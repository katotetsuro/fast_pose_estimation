import torch
import numpy as np

from pose_proposal_networks.coder import decode
from pose_proposal_networks.pose_proposal_network import PoseProposalNetwork as C


def ppn_loss(x, t):
    """
    x: torch.Tensor. predicted tensor. Shape should be (batch_size, 6(K+1)+H'W'L, H, W)
    t: np.array. concatenation of encoded tensor
    """
    assert x.shape == t.shape
    if x.is_cuda:
        t = t.cuda()
    batchsize = x.shape[0]
    gt_keypoints = t[:, :6*(C.num_keypoints+1)]
    gt_limbs = t[:, 6*(C.num_keypoints+1):]
    gt_resp = gt_keypoints[:, 0::6]

    keypoints = x[:, :6*(C.num_keypoints+1)]
    limbs = x[:, 6*(C.num_keypoints+1):]
    resp = (keypoints[:, 0::6] == 1).float()

    loss_resp = torch.sum((
        resp - gt_resp)**2) * 0.25 / batchsize

    # iou
    _, _, pred_y1, pred_x1, pred_y2, pred_x2 = decode(keypoints)
    area_pred = (pred_y2 - pred_y1) * (pred_x2 - pred_x1)

    _, _, gt_y1, gt_x1, gt_y2, gt_x2 = decode(gt_keypoints)
    area_gt = (gt_y2 - gt_y1) * (gt_x2 - gt_x1)

    x1 = torch.max(pred_x1, gt_x1)
    y1 = torch.max(pred_y1, gt_y1)
    x2 = torch.min(pred_x2, gt_x2)
    y2 = torch.min(pred_y2, gt_y2)
    area_intersection = (y2 - y1) * (x2 - x1) * ((x1 < x2) * (y1 < y2)).float()

    iou = area_intersection / (area_pred + area_gt - area_intersection + 1e-6)
    loss_iou = torch.sum(
        gt_resp * (iou - keypoints[:, 1::6])**2) / batchsize

    # coor
    pred_ox = keypoints[:, 2::6]
    pred_oy = keypoints[:, 3::6]
    gt_ox = gt_keypoints[:, 2::6]
    gt_oy = gt_keypoints[:, 3::6]
    loss_coor = torch.sum(gt_resp * ((pred_ox - gt_ox)**2 +
                                     (pred_oy - gt_oy)**2)) * 5 / batchsize

    # size
    pred_w = torch.sqrt(torch.clamp(keypoints[:, 4::6], 0.0, 1.0))
    pred_h = torch.sqrt(torch.clamp(keypoints[:, 5::6], 0.0, 1.0))
    gt_w = torch.sqrt(gt_keypoints[:, 4::6])
    gt_h = torch.sqrt(gt_keypoints[:, 5::6])
    loss_size = torch.sum(
        gt_resp * ((pred_w-gt_w)**2 + (pred_h-gt_h)**2)) * 5 / batchsize

    # limb 0.5
    # loss_limb = 0
    # for i in range(C.output_rows):
    #     for j in range(C.output_cols):
    #         for dx, dy in itertools.product(range(-4, 5), repeat=2):
    #             ind_y = i + dy
    #             ind_x = j + dx
    #             if ind_y < 0 or C.output_rows <= ind_y or ind_x < 0 or C.output_cols <= ind_x:
    #                 continue
    #             r1 = keypoints[:, 0::6, j, i]
    #             r2 = keypoints[:, 0::6, ind_x, ind_y]
    #             m_r = F.maximum(r1, r2)
    #             p_r = r1 * r2
    #             limbs[:, ]

    # loss_limb *= 0.5 / batchsize

    total_loss = loss_resp + loss_iou + loss_coor + loss_size

    return {
        'loss_resp': loss_resp,
        'loss_iou': loss_iou,
        'loss_coor': loss_coor,
        'loss_size': loss_size,
        'loss': total_loss
    }
