import chainer
import chainer.links as L
import chainer.functions as F
from chainercv.links.model.resnet import ResNet50

from pose_proposal_networks.constants import *


def build_backbone():
    res = ResNet50()
    res.pick = 'res5'
    return chainer.Sequential(
        lambda x: res(x),
        L.Convolution2D(None, 512, ksize=3, stride=1, pad=1),
        lambda x: F.leaky_relu(x, slope=0.1),
        L.Convolution2D(None, 512, ksize=3, stride=1, pad=1),
        lambda x: F.leaky_relu(x, slope=0.1),
        L.Convolution2D(None, 6*(num_keypoints+1)+Hp *
                        Wp*num_limbs, ksize=1, stride=1, pad=0)
    )
