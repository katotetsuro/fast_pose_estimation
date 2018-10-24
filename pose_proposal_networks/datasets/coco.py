import os
import torch.utils.data as data
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

from pose_proposal_networks.coder import encode


class CocoKeypoint(data.Dataset):
    """`MS Coco Keypoint <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, encode=True):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        person_ids = list(self.coco.getImgIds(catIds=[1]))  # person only
        # select only head, left_shoulder, and right_shoulder are present data.
        all_img_infos = [(i['file_name'], i['id'])
                         for i in self.coco.loadImgs(person_ids)]
        self.ids = []
        for info in all_img_infos:
            file_name, img_id = info
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

            # 登場人物全員の鼻、両肩が見えるデータに絞る
            visible_enough = []
            for ann in anns:
                k = ann['keypoints']
                visible_enough.append(k[0*3+2] == k[5*3+2] == k[6*3+2] == 2)

            if all(visible_enough):
                self.ids.append(img_id)

        self.transform = transform
        self.encode = encode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, person_bboxes, parts_bboxes). 
            person_bboxes is ndarray, its shape is (R, 4)
            R is number of person in image. 2nd axis is [y1, x1, y2, x2], 
            parts_bbox is ndarray, its shape is (R, 17, 5).
            R is number of person in image.
            17 is number of parts in coco.
            5 is y1, x1, y2, x2, visibility(visibile if 2)
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        kps = np.stack([np.asarray(ann['keypoints']).reshape(17, 3)
                        for ann in target])

        left_diffs = np.abs(kps[:, 5, :2] - kps[:, 0, :2])
        right_diffs = np.abs(kps[:, 6, :2] - kps[:, 0, :2])
        person_sizes = np.mean(left_diffs.reshape(
            (-1, 2)), axis=-1) + np.mean(right_diffs.reshape((-1, 2)), axis=-1)
        person_sizes = person_sizes.reshape((-1, 1))

        person_sizes *= 0.5
        parts_sizes = person_sizes * 0.5
        parts_sizes = parts_sizes.reshape((-1, 1, 1))

        person_bboxes = np.concatenate(
            [kps[:, 0, :2] - person_sizes, kps[:, 0, :2] + person_sizes], axis=-1)
        person_bboxes = person_bboxes[:, [1, 0, 3, 2]]
        parts_bboxes = np.concatenate(
            [kps[:, :, :2]-parts_sizes, kps[:, :, :2]+parts_sizes, kps[:, :, [2]]], axis=-1)
        parts_bboxes = parts_bboxes[:, :, [1, 0, 3, 2, 4]]

        if self.transform is not None:
            img, person_bboxes, parts_bboxes = self.transform(
                img, person_bboxes, parts_bboxes)

        if self.encode:
            encoded_boxes = encode(person_bboxes, parts_bboxes)
            return img, encoded_boxes
        else:
            # データの描画のために残しといたほうがいい？
            return img, person_bboxes, parts_bboxes

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
