"""Human segmentation loss classes"""
import torch  # type: ignore

from lib import *  # type: ignore


class HumanSegmentationLossDice(torch.nn.modules.loss._Loss):
    """
    Dice loss

    Parameters
    ----------
    masks_pred : list of np.ndarrays
        boolean 1-channel masks, predicted by the model

    masks : list of np.ndarrays
        boolean 1-channel ground-truth masks
    """
    def __init__(self):
        super(HumanSegmentationLossDice, self).__init__()

        self.dice_function = get_dice

    def __call__(self, masks_pred, masks):
        return self.forward(masks_pred, masks)

    def forward(self, masks_pred, masks):
        loss = self.dice_function(masks_pred, masks)

        return loss
