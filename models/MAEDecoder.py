import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Block 
from utils.pos_embed import get_2d_sincos_pos_embed

class StandardMAEDecoder(nn.Module):
    """标准的MAE解码器实现

    本解码器接收来自编码器的潜空间表示（latent representation），
    然后重建出完整的Token序列。这个序列可以被重塑和投影，
    以匹配专家模型输出的多尺度特征。
    """
    def __init__(self, 
                 # DINOv2 Backbone的配置信息
                 num_patches, 
                 encoder_dim, 
                 # 解码器自身的配置信息
                 decoder_embed_dim=512, 
                 decoder_depth=8, 
                 decoder_num_heads=16,
                 mlp_ratio=4., 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        """
        初始化函数

        Args:
            num_patches (int): 原始图片中的块（Patch）总数 (例如 3249)。
            encoder_dim (int): 来自DINOv2编码器的潜特征维度 (例如 768)。
            decoder_embed_dim (int): 解码器内部使用的嵌入维度。
            decoder_depth (int): 解码器中Transformer模块的层数。
            decoder_num_heads (int): 解码器中多头注意力的头数。
            mlp_ratio (float): Transformer模块中MLP层的维度缩放比例。
            norm_layer (nn.Module): 使用的归一化层。
        """
        super().__init__()

        self.num_patches = num_patches
        self.encoder_dim = encoder_dim
        self.decoder_embed_dim = decoder_embed_dim

        # 1. 线性嵌入层：将编码器输出的特征维度，投影到解码器自己的工作维度
        self.decoder_embed = nn.Linear(encoder_dim, decoder_embed_dim, bias=True)

        # 2. 可学习的 [MASK] Token
        # 当一个patch被遮盖时，我们就用这个token的向量来填充
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # 3. 解码器专用的位置编码
        # 这是固定的sin-cos编码，且与编码器的位置编码是独立的
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False)

        # 4. 解码器的核心：Transformer模块
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])

        # 5. 最终的归一化层和投影层
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # 6. 最终的预测头 (decoder_pred)
        # 官方实现是投影回像素值 (patch_size**2 * 3)
        # 我们的目标是重建特征，所以这里我们把它投影回编码器的维度 (encoder_dim)
        # 因为我们的"像素"就是DINOv2的特征
        self.decoder_pred = nn.Linear(decoder_embed_dim, encoder_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # 使用sin-cos函数初始化位置编码
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            int(self.num_patches**.5), 
            cls_token=False # 解码器的位置编码不需要为[CLS]token留位置
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # 初始化mask token和其他网络层
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def reshape_pred_tokens(self, pred_tokens):
        """
        (B, N, D) -> (B, D, H_patch, W_patch)
        """
        B = pred_tokens.shape[0]
        h_patch = w_patch = int(self.num_patches**0.5)
        # (B, N, D) -> (B, H_patch, W_patch, D)
        pred_feature_map = pred_tokens.reshape(B, h_patch, w_patch, self.encoder_dim)
        # (B, H_patch, W_patch, D) -> (B, D, H_patch, W_patch)
        F_reconstructed = pred_feature_map.permute(0, 3, 1, 2)
        
        return F_reconstructed

    def forward(self, latent_tokens, ids_restore):
        """
        解码器的前向传播

        Args:
            latent_tokens (torch.Tensor): 学生模型编码器的输出（只包含可见块）。
                                          形状: (B, len_keep, D_encoder)。
            ids_restore (torch.Tensor): 用于恢复原始块顺序的索引。
                                        形状: (B, N)。

        Returns:
            torch.Tensor: 为整个图片重建出的特征序列。
                          形状: (B, N, D_encoder)。
        """
        # 1. 将可见块的特征投影到解码器的工作维度
        x = self.decoder_embed(latent_tokens)

        # 2. 为被遮盖的位置创建mask tokens
        num_masked = ids_restore.shape[1] - x.shape[1]
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1)

        # 3. 将可见块的特征和mask tokens拼接起来
        # 注意：此时的序列还是被打乱的顺序
        x_full = torch.cat([x, mask_tokens], dim=1)  # 形状: (B, N, D_decoder)

        # 4. 使用ids_restore恢复序列的原始空间顺序
        x_restored = torch.gather(x_full, 
                                  dim=1, 
                                  index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim)
                                  )

        # 5. 为完整的序列添加解码器的位置编码
        x_restored = x_restored + self.decoder_pos_embed

        # 6. 将序列送入解码器的Transformer模块进行信息传播和重建
        for blk in self.decoder_blocks:
            x_restored = blk(x_restored)
        
        x_restored = self.decoder_norm(x_restored)

        # 7. 将重建好的序列投影回DINOv2的原始特征维度
        pred_tokens = self.decoder_pred(x_restored) # 形状: (B, N, D_encoder)

        return self.reshape_pred_tokens(pred_tokens)
