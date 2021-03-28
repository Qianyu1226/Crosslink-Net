import random
import numpy as np
from skimage.io import imread
from skimage.transform import resize


def random_horizontal_flipN(imgs, u=0.5):
    """
    random horiziontal flip
    :param imgs:
    :param u:
    :return:
    """
    if random.random() < u:
        for n, img in enumerate(imgs):
            imgs[n] = np.fliplr(img)
    return imgs


def resizeN(imgs, new_size):
    """
    resize input images
    :param imgs:
    :param new_size:
    :return:
    """
    for i, img  in enumerate(imgs):
        imgs[i] = resize(img, new_size, mode='reflect', preserve_range=True)
    return imgs