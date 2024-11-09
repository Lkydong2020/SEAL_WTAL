import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import torchvision
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device) 

def Process_consistency_loss(self, proposals, prop_attn, prop_ciou, device, mask):
        """
        Calculate the PC_loss based on proposals and pseudo-ground truth proposals.

        Inputs:
            proposals: list of [M, 2] tensors
            prop_attn: tensor of attention scores, shape [B, M, 1]
            prop_ciou: tensor of predicted IoU values, shape [B, M, 1]
            mask: binary mask tensor indicating valid attention scores, shape [B, M, 1]
            device: the device on which to perform calculations

        Outputs:
            total_loss: tensor
        """
        total_center_loss = 0.0
        total_normalized_center_loss = 0.0
        total_penalty_loss = 0.0
        valid_count = 0

        for b in range(len(proposals)):
            # Use mask to obtain valid attention scores and proposals
            attn_b = prop_attn[b][mask[b].squeeze(-1)]
            pred_iou_b = prop_ciou[b][mask[b].squeeze(-1)]
            proposal_b = torch.tensor(proposals[b], dtype=torch.float32).to(device) 

            if attn_b.numel() == 0:
                continue
            
            # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&Construct 'pseudo-true instance labels'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            # Find the maximum attention score and determine the threshold
            max_attention_score = torch.max(attn_b)
            attention_threshold = max_attention_score * 0.9

            # Construct 'pseudo-true instance labels' using candidate proposals with attention scores greater than the threshold
            retained_mask = attn_b >= attention_threshold
            p_gt_segment_tensor = proposal_b[retained_mask.squeeze(-1)]

            if p_gt_segment_tensor.shape[0] == 0:
                continue
            # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

            # Calculate the center position
            p_gt_centers = (p_gt_segment_tensor[:, 0] + p_gt_segment_tensor[:, 1]) / 2
            pred_centers = (proposal_b[:, 0] + proposal_b[:, 1]) / 2

            # calculate the IOU
            iou = self.segments_iou(proposal_b, p_gt_segment_tensor)
            max_iou, max_iou_indices = torch.max(iou, dim=1)
            valid_indices = max_iou > 0
            if valid_indices.sum() == 0:
                continue

            valid_proposals = proposal_b[valid_indices.nonzero(as_tuple=True)[0]]
            matched_p_gt_segments = p_gt_segment_tensor[max_iou_indices[valid_indices]]
            matched_p_gt_centers = p_gt_centers[max_iou_indices[valid_indices]]
            intersection = torch.min(valid_proposals[:, 1], matched_p_gt_segments[:, 1]) - torch.max(valid_proposals[:, 0], matched_p_gt_segments[:, 0])

            # Calculate the center distance loss
            center_diff = torch.abs(pred_centers[valid_indices] - matched_p_gt_centers)
            normalized_center_diff = center_diff / (intersection + 1e-6)
            pred_iou_selected = pred_iou_b[valid_indices.nonzero(as_tuple=True)[0]]

            s1 = torch.min(valid_proposals[:, 0], matched_p_gt_segments[:, 0])
            s2 = torch.max(valid_proposals[:, 0], matched_p_gt_segments[:, 0])
            e1 = torch.min(valid_proposals[:, 1], matched_p_gt_segments[:, 1])
            e2 = torch.max(valid_proposals[:, 1], matched_p_gt_segments[:, 1])

            pred_center_expr = (e1 - s2) / 2 * pred_iou_selected + (s1 + s2) / 2

            center_loss = F.smooth_l1_loss(pred_center_expr, matched_p_gt_centers.view(-1), reduction='sum')
            total_center_loss += center_loss

            normalized_center_diff_loss = F.smooth_l1_loss(normalized_center_diff, torch.zeros_like(normalized_center_diff), reduction='sum')
            total_normalized_center_loss += normalized_center_diff_loss

            # Calculate the penalty
            p_gt_lengths = matched_p_gt_segments[:, 1] - matched_p_gt_segments[:, 0]
            intersection_ratio = intersection / p_gt_lengths
            penalty_indices = intersection_ratio < 0.5
            penalty_loss = torch.sum((1 - pred_iou_selected[penalty_indices]))
            total_penalty_loss += penalty_loss

            valid_count += valid_indices.sum().item()

        if valid_count > 0:
            total_center_loss /= valid_count
            total_normalized_center_loss /= valid_count
            total_penalty_loss /= valid_count

        total_IOU_loss = total_center_loss + total_normalized_center_loss + total_penalty_loss
        return total_IOU_loss