import chainer
from pose_proposal_networks.loss import ppn_loss


class PPNTrainChain(chainer.Chain):
    def __init__(self, model):
        self.model = model

    def __call__(self, x, t):
        loss_resp, loss_iou, loss_coor, loss_size = ppn_loss(x, t)
        loss = loss_resp + loss_iou + loss_coor + loss_size

        chainer.reporter.report({
            'loss_resp': loss_resp,
            'loss_iou': loss_iou,
            'loss_coor': loss_coor,
            'loss_size': loss_size,
            'loss': loss
        }, self)

        return loss
