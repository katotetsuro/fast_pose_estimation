import torchvision
import torch.nn as nn


class PoseProposalNetwork(nn.Module):
    num_keypoints = 17

    """
    search window size for limbs
    """
    Hp = 9
    Wp = 9

    """
    from
    https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py#L49-L63
    """
    num_limbs = 15

    output_dim = 6 * (num_keypoints + 1) + Hp * Wp * num_limbs

    stride = 32
    input_width = 384
    output_cols = input_width // stride
    input_height = 384
    output_rows = input_height // stride

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(True)
        extractor = list(resnet.children())[:8]
        self.model = nn.Sequential(
            *extractor,
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, self.output_dim, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.model(x)

    def resize(self, img, person_bboxes=None, parts_bboxes=None):
        sy = img.height
        sx = img.width
        sx = self.input_width / sx
        sy = self.input_height / sy

        resized_img = torchvision.transforms.functional.resize(
            img, (self.input_width, self.input_height))

        if person_bboxes is None:
            return resized_img

        person_bboxes[:, 0::2] *= sy
        person_bboxes[:, 1::2] *= sx
        parts_bboxes[:, :, 0:-1:2] *= sy
        parts_bboxes[:, :, 1:-1:2] *= sx

        return resized_img, person_bboxes, parts_bboxes
