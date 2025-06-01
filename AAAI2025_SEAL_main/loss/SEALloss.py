import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init 

def prop_topk_loss(cas, labels, mask_cas, is_back=True, topk=8):
    """
    Compute the topk classification loss

    Inputs:
        cas: tensor of size [B, M, C]
        labels: tensor of size [B, C]
        mask_cas: tensor of size [B, M, C]
        is_back: bool
        topk: int

    Outputs:
        loss_mil: tensor
    """
    if is_back:
        labels_with_back = torch.cat((labels, torch.ones_like(labels[:, [0]])), dim=-1)
    else:
        labels_with_back = torch.cat((labels, torch.zeros_like(labels[:, [0]])), dim=-1)
    labels_with_back = labels_with_back / (torch.sum(labels_with_back, dim=-1, keepdim=True) + 1e-4)

    loss_mil = 0
    for b in range(cas.shape[0]):
        cas_b = cas[b][mask_cas[b]].reshape((-1, cas.shape[-1]))
        topk_val, _ = torch.topk(cas_b, k=max(1, int(cas_b.shape[-2] // topk)), dim=-2)
        video_score = torch.mean(topk_val, dim=-2)
        loss_mil += - (labels_with_back[b] * F.log_softmax(video_score, dim=-1)).sum(dim=-1).mean()
    loss_mil /= cas.shape[0]

    return loss_mil

def decay_weight(epoch, total_epochs, k):
    if epoch > total_epochs or total_epochs==0:
        return 0
    return np.exp(-k * epoch)

def SME_weight_decay2(t, total_iterations, alpha=0.05, beta=10):
    if t > total_iterations or total_iterations == 0:
        return 0
    t_normalized = t / total_iterations
    weight = np.exp(-alpha * t_normalized) * (1 - np.exp(-beta * t_normalized))
    return weight

def segments_iou(segments1, segments2):
    """
    Inputs:
        segments1: tensor of size [M1, 2]
        segments2: tensor of size [M2, 2]

    Outputs:
        iou_temp: tensor of size [M1, M2]
    """
    segments1 = segments1.unsqueeze(1)                          # [M1, 1, 2]
    segments2 = segments2.unsqueeze(0)                          # [1, M2, 2]
    tt1 = torch.maximum(segments1[..., 0], segments2[..., 0])   # [M1, M2]
    tt2 = torch.minimum(segments1[..., 1], segments2[..., 1])   # [M1, M2]
    intersection = tt2 - tt1
    union = (segments1[..., 1] - segments1[..., 0]) + (segments2[..., 1] - segments2[..., 0]) - intersection
    iou = intersection / (union + 1e-6)                         # [M1, M2]
    # Remove negative values
    iou_temp = torch.zeros_like(iou)
    iou_temp[iou > 0] = iou[iou > 0]
    return iou_temp

def PC_loss(proposals, prop_attn, prop_ciou, device, mask, args):
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
        if not torch.is_tensor(proposals[b]):
            proposal_b = torch.tensor(proposals[b], dtype=torch.float32, device=device)
        else:
            proposal_b = proposals[b].to(device)

        if attn_b.numel() == 0:
            continue

        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&Construct 'pseudo-true instance labels'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        # Find the maximum attention score and determine the threshold
        max_attention_score = torch.max(attn_b)
        attention_threshold = max_attention_score * args.gamma

        # Construct 'pseudo-true instance labels' using candidate proposals with attention scores greater than the threshold
        retained_mask = attn_b >= attention_threshold
        p_gt_segment_tensor = proposal_b[retained_mask.squeeze(-1)]

        if p_gt_segment_tensor.shape[0] == 0:
            continue

        # Calculate the center position
        p_gt_centers = (p_gt_segment_tensor[:, 0] + p_gt_segment_tensor[:, 1]) / 2
        pred_centers = (proposal_b[:, 0] + proposal_b[:, 1]) / 2

        # calculate the IOU
        iou = segments_iou(proposal_b, p_gt_segment_tensor)
        max_iou, max_iou_indices = torch.max(iou, dim=1)
        valid_indices = max_iou > 0
        if valid_indices.sum() == 0:
            continue

        valid_proposals = proposal_b[valid_indices.nonzero(as_tuple=True)[0]]
        matched_p_gt_segments = p_gt_segment_tensor[max_iou_indices[valid_indices]]
        matched_p_gt_centers = p_gt_centers[max_iou_indices[valid_indices]]
        intersection = torch.min(valid_proposals[:, 1], matched_p_gt_segments[:, 1]) - torch.max(valid_proposals[:, 0], matched_p_gt_segments[:, 0])
        intersection = torch.clamp(intersection, min=1e-6)

        # Calculate the center distance loss
        center_diff = torch.abs(pred_centers[valid_indices] - matched_p_gt_centers)
        normalized_center_diff = center_diff / (intersection)
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
        if penalty_indices.any():
            penalty_loss = (1 - pred_iou_selected[penalty_indices]).sum()
        else:
            penalty_loss = torch.tensor(0.0, device=device)
        total_penalty_loss += penalty_loss

        valid_count += valid_indices.sum().item()

    # Total_loss for PC_loss
    if valid_count > 0:
        total_center_loss /= valid_count
        total_normalized_center_loss /= valid_count
        total_penalty_loss /= valid_count

    total_IOU_loss = 0.01 * total_center_loss + args.alpha_5 * total_normalized_center_loss + args.alpha_6 * total_penalty_loss
    return total_IOU_loss

def SME_loss(prop_fused_cas, prop_mask, prop_attn, features, mask, appearance_descriptors, motion_descriptors, epoch, total_epochs, args):
    """
    Calculate the cross entropy loss with one-hot pseudo labels and decaying weight for inconsistent samples.

    Inputs:
        prop_fused_cas: tensor of shape [B, M, C], the predicted class scores after the Backbone_Proposal
        prop_mask: tensor of shape [B, M], indicating valid positions in the prop_fused_cas
        features: tensor of shape [B, T, 2048], original features before SME
        mask: tensor of shape [B, T], indicating valid positions in the features
        appearance_descriptors: tensor of shape [20, 1024], appearance descriptors
        motion_descriptors: tensor of shape [20, 1024], motion descriptors
        epoch: current training epoch
        total_epochs: total number of training epochs

    Outputs:
        loss: tensor representing the total loss
    """
    prop_fused_cas = prop_fused_cas[:, :, :-1]

    combined_descriptors = torch.cat((appearance_descriptors, motion_descriptors), dim=1)
    valid_indices_features = mask.bool()  # [B, T]

    pooled_features = []
    rgb_features = []
    flow_features = []

    for i in range(features.size(0)):  
        valid_rgb_feature = features[i][valid_indices_features[i], :1024]
        rgb_features.append(valid_rgb_feature.mean(dim=0))

        valid_flow_feature = features[i][valid_indices_features[i], 1024:]
        flow_features.append(valid_flow_feature.mean(dim=0))

    rgb_features = torch.stack(rgb_features)  # [B, 1024]
    flow_features = torch.stack(flow_features)  # [B, 1024]

    cos_sim_rgb = F.cosine_similarity(rgb_features.unsqueeze(1), appearance_descriptors.unsqueeze(0), dim=-1)  # [B, 20]
    cos_sim_flow = F.cosine_similarity(flow_features.unsqueeze(1), motion_descriptors.unsqueeze(0), dim=-1)  # [B, 20]

    cos_sim_rgb = F.softmax(cos_sim_rgb, dim=-1) 
    cos_sim_flow = F.softmax(cos_sim_flow, dim=-1)

    average_cos_sim = (cos_sim_rgb + cos_sim_flow) / 2  # [B, 20]
    _, pseudo_labels = torch.max(average_cos_sim, dim=1)
    one_hot_pseudo_labels = F.one_hot(pseudo_labels, num_classes=20).float()  # [B, 20]

    is_consistent = torch.argmax(cos_sim_rgb, dim=1) == torch.argmax(cos_sim_flow, dim=1)

    valid_indices_prop = prop_mask.bool()  # [B, M]
    pooled_prop_scores = []
    for i in range(prop_fused_cas.size(0)):  # [B, M, 20]
        valid_scores = prop_fused_cas[i][valid_indices_prop[i].squeeze(-1)]  # [M', 20]
        valid_attn = prop_attn[i][valid_indices_prop[i].squeeze(-1)]
        valid_attn = valid_attn.squeeze(-1)
        valid_scores = F.softmax(valid_scores, dim=-1)
        normalized_scores = valid_scores.mean(dim=0)  #  [20]
        pooled_prop_scores.append(normalized_scores)

    pooled_prop_scores = torch.stack(pooled_prop_scores)  # [B, 20]

    high_confidence_loss = F.mse_loss(pooled_prop_scores[is_consistent], one_hot_pseudo_labels[is_consistent])

    inconsistent_indices = ~is_consistent
    if inconsistent_indices.any():
        max_class_rgb = torch.argmax(cos_sim_rgb[inconsistent_indices], dim=1)
        max_class_flow = torch.argmax(cos_sim_flow[inconsistent_indices], dim=1)

        pseudo_label_indices = torch.where(
            (pseudo_labels[inconsistent_indices] == max_class_rgb) | 
            (pseudo_labels[inconsistent_indices] == max_class_flow)
        )

        if len(pseudo_label_indices[0]) > 0:
            consistent_pseudo_labels = one_hot_pseudo_labels[inconsistent_indices][pseudo_label_indices]
            consistent_pooled_scores = pooled_prop_scores[inconsistent_indices][pseudo_label_indices]

            decay_weight = 1.0 - (epoch / total_epochs)
            low_confidence_loss = F.mse_loss(consistent_pooled_scores, consistent_pseudo_labels) * decay_weight
        else:
            low_confidence_loss = 0.0
    else:
        low_confidence_loss = 0.0

    high_confidence_loss = high_confidence_loss.mean()
    if isinstance(low_confidence_loss, torch.Tensor):
        low_confidence_loss = low_confidence_loss.mean()

    # total_loss for SME_loss
    loss =  high_confidence_loss +  0.5 * low_confidence_loss 

    return loss


def SC_loss(prop_fused_cas, prop_cas_supp, prop_fused_feat_fuse, labels, epoch, total_epochs, args):
    """
    Calculate the simplified semantic consistency loss with dynamic weighting.

    Inputs:
        prop_fused_cas: tensor of shape [B, M, C], predicted class scores after the Backbone_Proposal
        prop_cas_supp: tensor of shape [B, M, C], background suppressed predicted class scores
        prop_fused_feat_fuse: tensor of shape [B, M, D], the fused features after the Backbone_Proposal
        labels: tensor of shape [B, C], video-level labels
        epoch: current training epoch
        total_epochs: total number of training epochs

    Outputs:
        loss: tensor representing the simplified semantic consistency loss
    """

    prop_fused_cas = prop_fused_cas[:, :, :-1]
    prop_cas_supp = prop_cas_supp[:, :, :-1]
    prop_fused_cas = F.softmax(prop_fused_cas, dim=-1)
    prop_cas_supp = F.softmax(prop_cas_supp, dim=-1)

    soft_descriptors = []
    batch_size, num_proposals, num_classes = prop_fused_cas.shape

    for i in range(batch_size):
        correct_class_mask = (torch.argmax(prop_fused_cas[i], dim=1) == torch.argmax(prop_cas_supp[i], dim=1)) & \
                            (torch.argmax(prop_fused_cas[i], dim=1) == torch.argmax(labels[i]))

        if correct_class_mask.sum().item() > 0:
            soft_descriptor = prop_fused_feat_fuse[i][correct_class_mask].mean(dim=0)
        else:
            soft_descriptor = torch.zeros(prop_fused_feat_fuse.size(-1)).to(prop_fused_cas.device)

        soft_descriptors.append(soft_descriptor)

    soft_descriptors = torch.stack(soft_descriptors)  # [B, D]

    semantic_consistency_loss = 0.0
    total_segments = 0  
    for i in range(batch_size):
        classification_mask = (torch.argmax(prop_fused_cas[i], dim=1) == torch.argmax(labels[i])) | \
                            (torch.argmax(prop_cas_supp[i], dim=1) == torch.argmax(labels[i]))

        num_segments = classification_mask.sum().item()
        if num_segments > 0:
            selected_features = prop_fused_feat_fuse[i][classification_mask]
            distance = torch.norm(selected_features - soft_descriptors[i].unsqueeze(0).expand_as(selected_features), p=2, dim=1)
            semantic_consistency_loss += distance.sum()
            total_segments += num_segments  

    if total_segments > 0:
        semantic_consistency_loss = semantic_consistency_loss/total_segments
    else:
        semantic_consistency_loss = torch.tensor(0.0, device = prop_fused_cas.device)

    consistency_loss = F.mse_loss(prop_fused_cas, prop_cas_supp)

    # Total_loss for SC_loss
    loss = (0.3 + 0.2 * (epoch / args.alpha_2)) * semantic_consistency_loss + consistency_loss

    return loss
