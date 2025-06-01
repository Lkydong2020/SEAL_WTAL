import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import torchvision
import os
import warnings
from utils import pad_features
from loss import (
    prop_topk_loss,
    decay_weight,
    SME_weight_decay2,
    PC_loss,
    SME_loss,
    SC_loss
)

warnings.filterwarnings("ignore", category=UserWarning)

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device) 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)

class Backbone_Proposal(torch.nn.Module):
    """
    Backbone for single modal in P-MIL framework
    """
    def __init__(self, feat_dim, n_class, dropout_ratio, roi_size):
        super().__init__()
        embed_dim = feat_dim // 2
        self.roi_size = roi_size

        self.prop_fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )
        self.prop_classifier = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, n_class+1, 1),
        )
        self.prop_attention = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 1),
        )
        self.prop_completeness = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 1),
        )

    def forward(self, feat):
        """
        Inputs:
            feat: tensor of size [B, M, roi_size, D]

        Outputs:
            prop_cas:  tensor of size [B, C, M]
            prop_attn: tensor of size [B, 1, M]
            prop_iou:  tensor of size [B, 1, M]
        """
        feat1 = feat[:, :,                   : self.roi_size//6  , :].max(2)[0]
        feat2 = feat[:, :, self.roi_size//6  : self.roi_size//6*5, :].max(2)[0]
        feat3 = feat[:, :, self.roi_size//6*5:                   , :].max(2)[0]
        feat = torch.cat((feat2-feat1, feat2, feat2-feat3), dim=2)

        feat_fuse = self.prop_fusion(feat)                              # [B, M, D]
        feat_fuse = feat_fuse.transpose(-1, -2)                         # [B, D, M]

        prop_cas = self.prop_classifier(feat_fuse)                      # [B, C, M]
        prop_attn = self.prop_attention(feat_fuse)                      # [B, 1, M]
        prop_iou = self.prop_completeness(feat_fuse)                    # [B, 1, M]

        return prop_cas, prop_attn, prop_iou, feat_fuse


class SEAL(torch.nn.Module):
    """
    PyTorch module for the SEAL framework!
    """
    def __init__(self, args, appearance_descriptors, motion_descriptors):
        super().__init__()
        n_class = args.num_class
        dropout_ratio = args.dropout_ratio
        self.feat_dim = args.feature_size
        self.max_proposal = args.max_proposal
        self.roi_size = args.roi_size
        
        self.prop_fused_backbone = Backbone_Proposal(self.feat_dim, n_class, dropout_ratio, self.roi_size)

        self.appearance_descriptors = appearance_descriptors
        self.motion_descriptors = motion_descriptors
        # self.apply(weights_init)

    def extract_roi_features(self, features, proposals, is_training):
        """
        Extract region of interest (RoI) features from raw i3d features based on given proposals

        Inputs:
            features: list of [T, D] tensors
            proposals: list of [M, 2] tensors
            is_training: bool

        Outputs:
            prop_features:tensor of size [B, M, roi_size, D]
            prop_mask: tensor of size [B, M]
        """
        num_prop = torch.tensor([prop.shape[0] for prop in proposals])
        batch, max_num = len(proposals), num_prop.max()
        # Limit the max number of proposals during training
        if is_training:
            max_num = min(max_num, self.max_proposal)
        prop_features = torch.zeros((batch, max_num, self.roi_size, self.feat_dim)).to(features[0].device)
        prop_mask = torch.zeros((batch, max_num)).to(features[0].device)

        for i in range(batch):
            feature = features[i]
            proposal = proposals[i]
            if num_prop[i] > max_num:
                sampled_idx = torch.randperm(num_prop[i])[:max_num]
                proposal = proposal[sampled_idx]

            # Extend the proposal by 25% of its length at both sides
            start, end = proposal[:, 0], proposal[:, 1]
            len_prop = end - start
            start_ext = start - 0.25 * len_prop
            end_ext = end + 0.25 * len_prop
            # Fill in blank at edge of the feature, offset 0.5, for more accurate RoI_Align results
            fill_len = torch.ceil(0.25 * len_prop.max()).long() + 1                         # +1 because of offset 0.5
            fill_blank = torch.zeros(fill_len, self.feat_dim).to(feature.device)
            feature = torch.cat([fill_blank, feature, fill_blank], dim=0)
            start_ext = start_ext + fill_len - 0.5
            end_ext = end_ext + fill_len - 0.5
            proposal_ext = torch.stack((start_ext, end_ext), dim=1)
            
            # Extract RoI features using RoI Align operation
            y1, y2 = proposal_ext[:, 0], proposal_ext[:, 1]
            x1, x2 = torch.zeros_like(y1), torch.ones_like(y2)
            boxes = torch.stack((x1, y1, x2, y2), dim=1)                                    # [M, 4]
            feature = feature.transpose(0, 1).unsqueeze(0).unsqueeze(3)                     # [1, D, T, 1]
            feat_roi = torchvision.ops.roi_align(feature, [boxes], [self.roi_size, 1])      # [M, D, roi_size, 1]
            feat_roi = feat_roi.squeeze(3).transpose(1, 2)                                  # [M, roi_size, D]
            prop_features[i, :proposal.shape[0], :, :] = feat_roi                           # [B, M, roi_size, D]
            prop_mask[i, :proposal.shape[0]] = 1                                            # [B, M]

        return prop_features, prop_mask

    def forward(self, features, proposals, is_training=True):
        """
        Inputs:
            features: list of [T, D] tensors
            proposals: list of [M, 2] tensors
            is_training: bool

        Outputs:
            outputs: dictionary
        """
        max_len = max([feat.size(0) for feat in features])
        padded_features, feature_masks = pad_features(features, max_len)
        features_ori = torch.stack(padded_features, dim=0).to(device)
        mask = torch.stack(feature_masks, dim=0).to(device)

        prop_features, prop_mask = self.extract_roi_features(features_ori, proposals, is_training)

        prop_fused_cas, prop_fused_attn, prop_fused_iou, prop_fused_feat_fuse = self.prop_fused_backbone(prop_features)

        outputs = {
            'prop_fused_cas': prop_fused_cas.transpose(-1, -2),                # [B, M, C]
            'prop_fused_attn': prop_fused_attn.transpose(-1, -2),              # [B, M, 1]
            'prop_fused_iou': prop_fused_iou.transpose(-1, -2),                # [B, M, 1]
            'prop_mask': prop_mask,                                            # [B, M]
            'features': features_ori,                                          # [B, T, 2048]
            'prop_fused_feat_fuse': prop_fused_feat_fuse.transpose(-1, -2),    # [B, D, M]
            "mask": mask
        }
        return outputs

    def criterion(self, outputs, labels, proposals, epoch, args, device):
        """
        Compute the total loss function

        Inputs: 
            outputs: dictionary
            labels: tensor of size [B, C]
            proposals: list of [M, 2] tensors
            epoch: int
            args: argparse.Namespace

        Outputs:
            loss_dict: dictionary
        """

        prop_fused_cas, prop_fused_attn, prop_fused_iou = outputs['prop_fused_cas'], outputs['prop_fused_attn'], outputs['prop_fused_iou']
        prop_mask = outputs['prop_mask']
        features = outputs['features']  
        prop_fused_feat_fuse = outputs['prop_fused_feat_fuse']
        mask = outputs['mask']

        prop_attn = torch.sigmoid(prop_fused_attn)                        # [B, M, 1]
        prop_iou = torch.sigmoid(prop_fused_iou)                          # [B, M, 1]
        prop_mask = prop_mask.unsqueeze(2).bool()                         # [B, M, 1]
        prop_mask_cas = prop_mask.repeat((1, 1, prop_fused_cas.shape[2])) # [B, M, C]

        # proposal classification loss
        prop_cas_supp = prop_fused_cas * prop_attn
        loss_prop_mil_orig = prop_topk_loss(prop_fused_cas, labels, prop_mask_cas, is_back=True, topk=args.k)
        loss_prop_mil_supp = prop_topk_loss(prop_cas_supp, labels, prop_mask_cas, is_back=False, topk=args.k)

        # SME_loss
        sme_loss = SME_loss(prop_fused_cas, prop_mask, prop_attn, features, mask, self.appearance_descriptors, self.motion_descriptors, epoch, args.max_epoch, args)
        
        # PC_loss
        pc_loss = PC_loss(proposals, prop_attn, prop_iou, device, prop_mask, args)

        # SC_loss
        sc_loss = SC_loss(prop_fused_cas, prop_cas_supp, prop_fused_feat_fuse, labels, epoch, args.max_epoch, args)

        loss_prop_mil_orig = 1 * loss_prop_mil_orig
        loss_prop_mil_supp = 0.5 * loss_prop_mil_supp

        loss_total = loss_prop_mil_orig + loss_prop_mil_supp + SME_weight_decay2(epoch, args.alpha_3) * sme_loss + args.alpha_1 * pc_loss + decay_weight(epoch, args.alpha_2, k=1/args.alpha_4) * sc_loss

        loss_dict = {
            'loss_total': loss_total,
            'loss_prop_mil_orig': loss_prop_mil_orig,
            'loss_prop_mil_supp': loss_prop_mil_supp,
            'SME_loss': sme_loss,
            'PC_loss': pc_loss,
            'SC_loss': sc_loss
        }
        return loss_dict
        

    