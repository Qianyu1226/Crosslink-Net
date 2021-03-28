from skimage.io import imread, imsave
from utils.visualize import visulize_gt
from glob import glob
import os
from skimage.measure import regionprops, label
import numpy as np

def get_coordinate(mask):
    areas = regionprops(label(mask//255))
    assert len(areas) == 1
    centroid = np.rint(areas[0].centroid)
    return centroid.astype(np.int)


def visualize(data_dir):
    os.makedirs(data_dir + 'result_vis_result/', exist_ok=True)
    paths = glob(data_dir + '*.jpg')
    print(paths)
    names = [name.split('.')[1].split('/')[-1] for name in paths]
    print(names)
    for name in names:

        img = imread(data_dir + name + '.jpg', as_grey=True)
        gt = imread(data_dir + name + '_1.bmp', as_grey=True)
        coordinate = get_coordinate(gt)

        img_cropped = img[coordinate[0] - 75:coordinate[0] + 75, coordinate[1] - 75:coordinate[1] + 75]
        imsave(data_dir + 'result_vis_result/' + name + '_cropped.eps', img_cropped)

        prediction = imread(data_dir + name + '.bmp', as_grey=True)

        vis_img = visulize_gt(img, gt)
        vis_img = visulize_gt(vis_img, prediction, mode='g')
        vis_img = vis_img[coordinate[0] - 75:coordinate[0] + 75, coordinate[1] - 75:coordinate[1] + 75]
        imsave(data_dir + 'result_vis_result/' + name + '.eps', vis_img)

if __name__ == '__main__':
    visualize('./result_vis/')