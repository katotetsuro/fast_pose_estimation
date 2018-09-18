import chainer
import chainer.functions as F
import pose_proposal_networks.constants as C
import itertools


def find_keypoint_size(keypoints):
    return 10

    # if upperbody_is_present:
    #     pass
    # elif face_segment_is_present:
    #     pass
    # else:
    #     return 10


def encode(gt_bboxes, gt_keypoints):
    """
    encode bboxes in one image.
    encoded tensor has fixed shape regardress of R, (6(K+1)+H'W'L, H, W)
    (so we can use concat_example)
    gt_bboxes (R, 4) R is number of bbox in this image
    gt_keypoints (R, num_keypoints, 3)
    """
    xp = chainer.backends.cuda.get_array_module(gt_bboxes)
    t = xp.zeros((C.output_dim, C.output_rows,
                  C.output_cols), dtype=xp.float32)

    ind_responsibility, ind_loc_conf, ind_ox, ind_oy, ind_w, ind_h = range(6)

    for box, keypoints in zip(gt_bboxes, gt_keypoints):
        # box part
        cy, cx = (box[:2] + box[2:]) * 0.5
        row = cy // C.stride
        col = cx // C.stride
        # center of the bounding box relative to the bounds of the grid cell
        #  with the scale normalized by the length of the cells
        oy = (cy - row * C.stride) / C.stride
        ox = (cx - col * C.stride) / C.stride
        h, w = box[2:] - box[:2]
        h /= C.input_height
        w /= C.input_width
        # ind in [0, ...., K]
        ind = 0
        # responsibility
        t[6*ind+ind_responsibility, row, col] = 1
        # IoU(no need to preprare)

        # oy, ox, h, w
        t[6*ind+ind_ox, row, col] = ox
        t[6*ind+ind_oy, row, col] = oy
        t[6*ind+ind_w, row, col] = w
        t[6*ind+ind_h, row, col] = h

        # keyponts part
        kp_box_size = find_keypoint_size(keypoints)
        for kp in keypoints:
            if kp[2] != 2:
                continue
            cy, cx = (box[:2] + box[2:]) * 0.5
            row = cy // C.stride
            col = cx // C.stride
            # center of the bounding box relative to the bounds of the grid cell
            #  with the scale normalized by the length of the cells
            oy = (cy - row * C.stride) / C.stride
            ox = (cx - col * C.stride) / C.stride
            h, w = box[2:] - box[:2]
            h /= C.input_height
            w /= C.input_width
            # ind in [0, ...., K]
            ind = 0
            # responsibility
            t[6*ind+ind_responsibility, row, col] = 1
            # IoU(no need to preprare)

            # oy, ox, h, w
            t[6*ind+ind_ox, row, col] = ox
            t[6*ind+ind_oy, row, col] = oy
            t[6*ind+ind_w, row, col] = w
            t[6*ind+ind_h, row, col] = h

        # limbs

    return t


def ppn_loss(x, t):
    """
    x: chainer.Variable. predicted tensor. Shape should be (batch_size, 6(K+1)+H'W'L, H, W)
    t: xp.array. concatenation of encoded tensor
    """
    assert x.shape == t.shape
    batchsize = x.shape[0]
    gt_keypoints = t[:, :6*(C.num_keypoints+1)]
    gt_limbs = t[:, 6*(C.num_keypoints+1):]
    gt_resp = gt_keypoints[:, 0::6]

    keypoints = x[:, :6*(C.num_keypoints+1)]
    limbs = x[:, 6*(C.num_keypoints+1):]
    resp = keypoints[:, 0::6]

    loss_resp = F.sum(F.squared_difference(resp, gt_resp)) * 0.25 / batchsize

    # iou
    xp = chainer.backends.cuda.get_array_module(x)

    def decode(ox, oy, w, h):
        ox = (ox + xp.arange(C.output_cols,
                             dtype=xp.float32).reshape((1, -1))) * C.stride
        oy = (oy + xp.arange(C.output_rows,
                             dtype=xp.float32).reshape((-1, 1))) * C.stride

        w = F.clip(w, 0.0, 1.0) * C.input_width
        h = F.clip(h, 0.0, 1.0) * C.input_height

        # decode
        x1 = F.clip(ox - w * 0.5, 0.0, C.input_width)
        x2 = F.clip(ox + w * 0.5, 0.0, C.input_width)
        y1 = F.clip(oy - h * 0.5, 0.0, C.input_height)
        y2 = F.clip(oy + h * 0.5, 0.0, C.input_height)

        return x1, x2, y1, y2

    pred_ox = keypoints[:, 2::6]
    pred_oy = keypoints[:, 3::6]
    pred_w = keypoints[:, 4::6]
    pred_h = keypoints[:, 5::6]
    pred_x1, pred_x2, pred_y1, pred_y2 = decode(
        pred_ox, pred_oy, pred_w, pred_h)
    area_pred = (pred_y2 - pred_y1) * (pred_x2 - pred_x1)

    gt_ox = gt_keypoints[:, 2::6]
    gt_oy = gt_keypoints[:, 3::6]
    gt_w = gt_keypoints[:, 4::6]
    gt_h = gt_keypoints[:, 5::6]
    gt_x1, gt_x2, gt_y1, gt_y2 = decode(gt_ox, gt_oy, gt_w, gt_h)
    area_gt = (gt_y2 - gt_y1) * (gt_x2 - gt_x1)

    x1 = F.maximum(pred_x1, gt_x1)
    y1 = F.maximum(pred_y1, gt_y1)
    x2 = F.minimum(pred_x2, gt_x2)
    y2 = F.minimum(pred_y2, gt_y2)
    area_intersection = (y2 - y1) * (x2 - x1)

    iou = area_intersection / (area_pred + area_gt - area_intersection)
    loss_iou = F.sum(gt_resp * F.square(iou - keypoints[:, 1::6])) / batchsize

    # coor
    pred_ox = keypoints[:, 2::6]
    pred_oy = keypoints[:, 3::6]
    gt_ox = gt_keypoints[:, 2::6]
    gt_oy = gt_keypoints[:, 3::6]
    loss_coor = F.sum(gt_resp * (F.square(pred_ox - gt_ox) +
                                 F.square(pred_oy - gt_oy))) * 5 / batchsize

    # size
    pred_w = F.sqrt(F.clip(keypoints[:, 4::6], 0.0, 1.0))
    pred_h = F.sqrt(F.clip(keypoints[:, 5::6], 0.0, 1.0))
    gt_w = xp.sqrt(gt_keypoints[:, 4::6])
    gt_h = xp.sqrt(gt_keypoints[:, 5::6])
    loss_size = F.sum(
        gt_resp * (F.square(pred_w-gt_w) + F.square(pred_h-gt_h))) * 5 / batchsize

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

    return loss_resp, loss_iou, loss_coor, loss_size
