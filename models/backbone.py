import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from utils import NestedTensor, is_main_process
from models.projector import MultiScaleProjector

from transformers import AutoBackbone
from peft import get_peft_model, LoraConfig



size_to_feature_extraction_layers = {
    "base": [2, 5, 8, 11],
    "large": [5, 11, 17, 23],
}


class DINOv2Backbone(nn.Module):
    def __init__(self, args, peft=False):
        super().__init__()
        dinov2_model_name = args.backbone 
        size = dinov2_model_name.split('-')[-1]
        self.feature_extraction_layers = size_to_feature_extraction_layers[size]
        self.projector_scale = args.projector_scale
        self.dinov2 = AutoBackbone.from_pretrained(
                dinov2_model_name,
                out_features=[f"stage{i}" for i in self.feature_extraction_layers],
                output_attentions=False, 
                return_dict=True  
            )
        
        config = self.dinov2.config
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        
        if peft:
            # 创建LoraConfig
            peft_config = LoraConfig(
                r=args.R_LoRA,  # LoRA的秩  
                lora_alpha=args.R_LoRA,
                use_dora=True,
                target_modules=["query", "key", "value"],
                lora_dropout=0.1,
                bias="none"
            )
            # 应用PEFT
            self.dinov2 = get_peft_model(self.dinov2, peft_config)
            if is_main_process():
                print("PEFT model created. Trainable parameters:")
                self.dinov2.print_trainable_parameters()
        else:
            for param in self.dinov2.parameters():
                param.requires_grad = False


        self.strides = [8, 16, 32, 64]
        self.num_channels = [self.hidden_size] * len(self.feature_extraction_layers)
        

        
        self.projector = MultiScaleProjector(
            in_channels=self.num_channels,
            out_channels=256,
            scale_factors=self.projector_scale,
            layer_norm=False,
            rms_norm=False,
        ) if peft else None


    
    def forward(self, tensor_list: NestedTensor):
        
        x = tensor_list.tensors
        
        outputs = self.dinov2(x)
        feats = list(outputs.feature_maps)
        feats = self.projector(feats) if hasattr(self, 'projector') else feats

        out: List[NestedTensor] = []
        for i, feat in enumerate(feats):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out
    
    @torch.no_grad()
    def forward_mae_teacher(self, tensor_list: NestedTensor):
        """
        专门为MAE任务设计的前向传播方法。
        它处理一个批次的完整图像，并返回编码器的输出特征。

        Args:
            tensor_list (NestedTensor): 输入的完整图片张量, 包含图像和掩码。

        Returns:
            out (torch.Tensor): 编码器对可见块处理后的输出特征。
        """
        x = tensor_list.tensors
        outputs = self.dinov2(x)
        return outputs.feature_maps

    def forward_mae(self, images, mask_ratio=0.75):
        """
        专门为MAE任务设计的前向传播方法。
        它处理一个批次的完整图像，在内部进行掩码，并只通过编码器处理可见部分。

        Args:
            images (torch.Tensor): 输入的完整图片张量, 形状 (B, C, H, W)。
            mask_ratio (float): 掩码比例。

        Returns:
            tuple:
                - latent (torch.Tensor): 编码器对可见块处理后的输出特征。
                - mask (torch.Tensor): 用于重建的二进制掩码。
                - ids_restore (torch.Tensor): 用于恢复块顺序的索引。
        """

        B, _, H, W = images.shape
        device = images.device
        
        # 1. 获取嵌入层输出 
        outputs = self.dinov2(images, output_hidden_states=True, return_dict=True)
        all_tokens_embedded = outputs.hidden_states[0]  # 第0个hidden_state就是嵌入层输出 (B, N+1, D)

        # 2. 分离 [CLS] token 和图像块
        patch_tokens = all_tokens_embedded[:, 1:, :]  # (B, N, D)
        num_patches = patch_tokens.shape[1]

        # 3. 生成掩码 
        len_keep = int(num_patches * (1 - mask_ratio))
        noise = torch.rand(B, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        # 4. 选择可见块
        visible_patches_embedded = torch.gather(
            patch_tokens, 
            dim=1, 
            index=ids_keep.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )

        # 5. 将可见块送入编码器
        encoder_outputs = self.dinov2.base_model.model.encoder(visible_patches_embedded, output_hidden_states=True, return_dict=True)
        latent = [encoder_outputs.hidden_states[i+1] for i in self.feature_extraction_layers]  # (B, len_keep, D)

        # 6. 生成二进制掩码
        mask = torch.ones(B, num_patches, device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return latent, mask, ids_restore


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for x in xs:
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos
