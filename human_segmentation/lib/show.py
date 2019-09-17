import matplotlib.pyplot as plt

from lib.dataset import *
from lib.html import *
from lib.loss import *
from lib.metrics import *
from lib.model import *
# from lib.show import *
from lib.trainer import *
from lib.unet import *
from lib.utils import *


def show_img_with_mask(img, mask, figsize=(14, 8)):
    """Shows image and mask.

    Parameters
    ----------
    img : np.ndarray
        Image.
    mask : np.ndarray
        Mask.
    figsize : tuple of 2 int, optional (default=(14, 8))
        Figure size.

    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(img)
    ax2.imshow(mask)
    ax1.axis("off")
    ax2.axis("off")
    plt.show()
