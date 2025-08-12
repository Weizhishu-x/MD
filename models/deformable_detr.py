# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from models.MAEDecoder import StandardMAEDecoder
from models.backbone import DINOv2Backbone
from utils.misc_utils import NestedTensor, nested_tensor_from_tensor_list

    
    
class DeformableDETR(nn.Module):
    """ This is the Group DETR v3 module that performs object detection """
    def __init__(self,
                 args,
                 backbone,
                 transformer,
                 num_classes,
                 num_queries,
                 num_feature_levels,
                 aux_loss=True,
                 group_detr=1,
                 two_stage=True,
                 lite_refpoint_refine=True,
                 bbox_reparam=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            group_detr: Number of groups to speed detr training. Default is 1.
            lite_refpoint_refine: TODO
        """
        super().__init__()
        self.args = args
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(self.hidden_dim, num_classes)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)

        query_dim=4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, self.hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        # iter update
        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # two_stage
        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)])
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)])
        
        # --- 实例化 MAEDecoder ---
        self.MAEDecoder = None
        self.backbone_freezed = None
        # --- 实例化判别器 ---
        self.dis_enc = None
        self.dis_dec = None
        self.GRL = None

    def build_MAEDecoder(self, img_size, device, depth=8, num_heads=16):
        encoder_dim = self.backbone[0].hidden_size
        patch_size = self.backbone[0].patch_size
        num_patches = (img_size // patch_size) ** 2
        self.MAEDecoder = StandardMAEDecoder(
            num_patches=num_patches,
            encoder_dim=encoder_dim,
            decoder_embed_dim=512, 
            decoder_depth=depth,
            decoder_num_heads=num_heads
        ).to(device)
        self.backbone_freezed = DINOv2Backbone(self.args, peft=False).to(device)

    def build_discriminator(self, device):
        if self.dis_enc is None and self.dis_dec is None:
            self.dis_enc = nn.ModuleList([
                MultiConv2d(self.hidden_dim, self.hidden_dim, 2, 3)
                for _ in range(self.num_feature_levels)
            ]).to(device)
            self.dis_dec = MLP(self.hidden_dim, self.hidden_dim, 2, 3).to(device)
    
    def discriminator_forward(self, srcs, hs):
        def apply_dis(memory, discriminator):
            return discriminator(grad_reverse(memory))
        
        outputs_domains_enc = []
        for lvl, src in enumerate(srcs):
            outputs_domains_enc.append(apply_dis(src, self.dis_enc[lvl]))
        outputs_domains_dec = apply_dis(hs, self.dis_dec)

        return outputs_domains_enc, outputs_domains_dec
    
    def reinitialize_detection_head(self, num_classes):
        # Create new classification head
        del self.class_embed
        self.add_module("class_embed", nn.Linear(self.transformer.d_model, num_classes))
        
        # Initialize with focal loss bias adjustment
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        if self.two_stage:
            del self.transformer.enc_out_class_embed
            self.transformer.add_module("enc_out_class_embed", nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(self.group_detr)]))

    def forward(self, samples: NestedTensor, enable_mae: bool = False, mask_ratio: float = 0.75):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            # only use one group in inference
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight)

        if self.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(hs)
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat(
                [outputs_coord_cxcy, outputs_coord_wh], dim=-1
            )
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

        outputs_class = self.class_embed(hs)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc.append(cls_enc_gidx)
            cls_enc = torch.cat(cls_enc, dim=1)
            out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}

        if self.training:
            if self.MAEDecoder is not None and self.backbone_freezed is not None and enable_mae:
                mae_features, mask_mae, ids_restore = self.backbone[0].forward_mae(samples.tensors, mask_ratio)
                with torch.no_grad():
                    F_teacher = self.backbone_freezed.forward_mae_teacher(samples)
                    # F_teacher = self.backbone[0].projector(F_teacher)  
                F_reconstructed = [self.MAEDecoder(mae_feature, ids_restore) for mae_feature in mae_features]
                # F_reconstructed = self.backbone[0].projector(F_reconstructed) 
                out['mae_output'] = {
                    'mae_reconstructed': F_reconstructed,
                    'mae_teacher': F_teacher,
                    'ids_restore': ids_restore,
                    'mask': mask_mae
                }
        
            if self.dis_enc is not None and self.dis_dec is not None:
                outputs_domains_enc, outputs_domains_dec = self.discriminator_forward(srcs, hs[-1])
                out['da_output'] = {
                    'domain_enc_all': outputs_domains_enc,
                    'domain_dec_all': outputs_domains_dec
                }
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        """ """
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        for i in range(vit_encoder_num_layers):
            if hasattr(self.backbone[0].encoder, 'blocks'): # Not aimv2
                if hasattr(self.backbone[0].encoder.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.blocks[i].drop_path.drop_prob = dp_rates[i]
            else: # aimv2
                if hasattr(self.backbone[0].encoder.trunk.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.trunk.blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MultiConv2d(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv2d(n, k, kernel_size=(3, 3), padding=1) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eta=1.0):
        ctx.eta = eta
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.eta), None


def grad_reverse(x, eta=1.0):
    return GradReverse.apply(x, eta)