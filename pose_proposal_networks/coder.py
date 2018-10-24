import torch
import numpy as np

from pose_proposal_networks.pose_proposal_network import PoseProposalNetwork as C


def encode(gt_person_bboxes, gt_parts_bboxes):
    """
    encode bboxes in one image.
    encoded tensor has fixed shape regardress of R, (6(K+1)+H'W'L, H, W)
    (so we can concat examples)
    gt_person_bboxes (R, 4) R is number of bbox in this image
    gt_parts_bboxes (R, num_keypoints, 3)
    """
    t = torch.zeros(C.output_dim, C.output_rows,
                    C.output_cols)

    ind_responsibility, ind_loc_conf, ind_ox, ind_oy, ind_w, ind_h = range(6)

    # cy, cx = (gt_person_bboxes[:, :2] + gt_parts_bboxes[:, 2:]) * 0.5
    # rows = (cy // C.stride).astype(np.int32)
    # cols = (cx // C.stride).astype(np.int32)
    # oy = (cy - rows * C.stride) / C.stride
    # ox = (cx - cols * C.stride) / C.stride
    # heights = gt_person_bboxes[:, 2] - gt_person_bboxes[:, 0]
    # heights /= C.input_height
    # widths = gt_person_bboxes[:, 3] - gt_person_bboxes[:, 1]
    # width /= C.input_width
    # rows, colsの場所に上記値を入れたい

    for box in gt_person_bboxes:
        cy, cx = (box[:2] + box[2:]) * 0.5
        row = int(cy // C.stride)
        col = int(cx // C.stride)
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
        t[6*ind+ind_responsibility, row, col] += 1
        # IoU(no need to preprare)

        # oy, ox, h, w
        t[6*ind+ind_ox, row, col] = ox
        t[6*ind+ind_oy, row, col] = oy
        t[6*ind+ind_w, row, col] = w
        t[6*ind+ind_h, row, col] = h

    for parts_per_person in gt_parts_bboxes:
        for i, box in enumerate(parts_per_person):
            if box[4] != 2:
                continue
            ind = i+1
            cy, cx = (box[:2] + box[2:4]) * 0.5
            row = int(cy // C.stride)
            col = int(cx // C.stride)
            # center of the bounding box relative to the bounds of the grid cell
            #  with the scale normalized by the length of the cells
            oy = (cy - row * C.stride) / C.stride
            ox = (cx - col * C.stride) / C.stride
            h, w = box[2:4] - box[:2]
            h /= C.input_height
            w /= C.input_width

            # responsibility
            t[6*ind+ind_responsibility, row, col] += 1
            # IoU(no need to preprare)

            # oy, ox, h, w
            t[6*ind+ind_ox, row, col] = ox
            t[6*ind+ind_oy, row, col] = oy
            t[6*ind+ind_w, row, col] = w
            t[6*ind+ind_h, row, col] = h

    return t


def decode(x):
    """
    decode network's output tensor to bbox
    assumes batch input
    """
    x = x[:, :6*(C.num_keypoints+1)]
    resp = x[:, 0::6]
    conf = x[:, 1::6]
    ox = x[:, 2::6]
    oy = x[:, 3::6]
    w = x[:, 4::6]
    h = x[:, 5::6]

    ox = (ox + torch.arange(C.output_cols,
                            dtype=torch.float32).reshape((1, -1))) * C.stride
    oy = (oy + torch.arange(C.output_rows,
                            dtype=torch.float32).reshape((-1, 1))) * C.stride

    w = torch.clamp(w, 0.0, 1.0) * C.input_width
    h = torch.clamp(h, 0.0, 1.0) * C.input_height

    x1 = torch.clamp(ox - w * 0.5, 0.0, C.input_width)
    x2 = torch.clamp(ox + w * 0.5, 0.0, C.input_width)
    y1 = torch.clamp(oy - h * 0.5, 0.0, C.input_height)
    y2 = torch.clamp(oy + h * 0.5, 0.0, C.input_height)

    return resp, conf, y1, x1, y2, x2
