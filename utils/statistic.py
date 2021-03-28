def dice_loss(masks, labels, is_average=True):
    """
    dice loss
    :param masks:
    :param labels:
    :return:
    """
    num = labels.size(0)
    m1 = masks.view(num, -1)
    m2 = labels.view(num, -1)
    intersection = (m1 * m2)
    score = 2 * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)

    if is_average:
        return score.sum() /num
    else:
        return score