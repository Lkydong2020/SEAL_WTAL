import os
import random
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F


def setup_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_options(args, parser, exp_dir):
    """ 
    Print and save options.
    It will print both current options and default values (if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options of {} ---------------\n'.format(args.model_name)
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the file
    file_name = os.path.join(exp_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def write_to_file(exp_dir, dmap, cmap, test_acc, epoch):
    """
    Write experiment results to a file.
    """
    fid = open(os.path.join(exp_dir, "mAP-results.log"), "a+")
    string_to_write = '{:4}'.format(epoch)
    for item in dmap:
        string_to_write += "  " + "%5.2f" % (item*100)
    string_to_write += "  " + "%5.2f" % (np.mean(dmap[:5])*100)
    string_to_write += "  " + "%5.2f" % (np.mean(dmap[2:])*100)
    string_to_write += "  " + "%5.2f" % (np.mean(dmap[:7])*100)
    string_to_write += "  " + "%5.2f" % (cmap)
    string_to_write += "  " + "%5.2f" % (test_acc)
    fid.write(string_to_write + "\n")
    fid.close()


def collate_fn(batch):
    """
    Collate function for creating batches of data samples.
    """
    keys = batch[0].keys()
    data = {key: [] for key in keys}
    for sample in batch:
        for key in keys:
            data[key].append(sample[key])
    return data

def pad_features(features, max_len):
    padded_features = []
    feature_masks = []
    for feature in features:
        pad_size = max_len - feature.size(0)
        if pad_size > 0:
            padding = torch.zeros(pad_size, feature.size(1)).to(feature.device)
            padded_feature = torch.cat([feature, padding], dim=0)
            mask = torch.cat([torch.ones(feature.size(0)), torch.zeros(pad_size)]).to(feature.device)
        else:
            padded_feature = feature
            mask = torch.ones(feature.size(0)).to(feature.device)
        padded_features.append(padded_feature)
        feature_masks.append(mask)
    return padded_features, feature_masks

# for THUMOS14 and ActivityNet 1.2/1.3 
def strlist2multihot(strlist, classlist):
    """
    Convert a list of label strings to a multihot label encoding.
    """
    multihot = np.zeros(len(classlist))
    ind = [list(classlist).index(s.encode("utf-8")) for s in strlist]
    multihot[ind] = 1
    return multihot

def soft_nms(dets, iou_thr=0.7, method='gaussian', sigma=0.1):
    """
    Apply Soft NMS to a set of detection results.
    """
    # expand dets with areas, and the second dimension is
    # x1, x2, label, score, area
    dets = np.array(dets)
    areas = dets[:, 1] - dets[:, 0] + 1
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 3], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1].tolist())

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        xx2 = np.minimum(dets[0, 1], dets[1:, 1])
        inter = np.maximum(xx2 - xx1 + 1, 0.0)
        iou = inter / (dets[0, -1] + dets[1:, -1] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 3] *= weight
        dets = dets[1:, :]

    return retained_box

def filter_segments(segment_predict, vn, ambilist):
    """
    Filter out segments overlapping with ambiguous_test segments.
    """
    num_segment = len(segment_predict)
    ind = np.zeros(num_segment)
    for i in range(num_segment):
        for a in ambilist:
            if a[0] == vn:
                gt = range(int(round(float(a[2]) * 25 / 16)), int(round(float(a[3]) * 25 / 16)))
                pd = range(int(segment_predict[i][0]), int(segment_predict[i][1]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(len(set(gt).union(set(pd))))
                if IoU > 0:
                    ind[i] = 1
    s = [segment_predict[i, :] for i in range(num_segment) if ind[i] == 0]
    return np.array(s)

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

@torch.no_grad()
def get_prediction_fused(vid_name, data_dict, dataset, args):
    """
    Return predictions for a given video.
    """
    prop_cas = data_dict['prop_fused_cas'][0]
    prop_attn = data_dict['prop_fused_attn'][0]
    prop_iou = data_dict['prop_fused_iou'][0]
    idx_all = list(dataset.videonames).index(vid_name)
    proposals = dataset.proposals[idx_all]

    prop_cas = F.softmax(prop_cas, dim=1)
    prop_attn = torch.sigmoid(prop_attn)
    prop_iou = torch.sigmoid(prop_iou)

    prop_score = prop_cas * prop_attn * prop_iou
    prop_score = prop_score.cpu().numpy()

    pred = np.where(pred_vid_score >= args.threshold_cls)[0]
    if len(pred) == 0:
        pred = np.array([np.argmax(pred_vid_score)])

    proposal_dict = {}
    for c in pred:
        c_temp = []
        for i in range(proposals.shape[0]):
            c_score = prop_score[i, c]
            c_temp.append([proposals[i, 0], proposals[i, 1], c, c_score])
        proposal_dict[c] = c_temp

    # soft-NMS
    final_proposals = []
    for class_id in proposal_dict.keys():
        temp_proposal = soft_nms(proposal_dict[class_id])
        final_proposals += temp_proposal

    # filter out the Ambiguous
    final_proposals = np.array(final_proposals)
    final_proposals = filter_segments(final_proposals, vid_name.decode(), dataset.ambilist)

    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(final_proposals)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(final_proposals[i, 0])
        t_end_lst.append(final_proposals[i, 1])
        label_lst.append(final_proposals[i, 2])
        score_lst.append(final_proposals[i, 3])

    prediction = pd.DataFrame({"video-id": video_lst,
                               "t-start": t_start_lst,
                               "t-end": t_end_lst,
                               "label": label_lst,
                               "score": score_lst,})
    return prediction, pred, pred_vid_score
