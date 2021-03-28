import torch
#import numpy as np
from glob import glob
#from skimage.io import imread, imsave
from torch.utils.data import Dataset, DataLoader
from dataset.transforms import *
from os.path import exists, join, abspath
import os

def img_to_tensor(img, mean=0, std=1):
    img = np.expand_dims(img, axis=0)
    assert img.shape[0] == 1

    img = img.astype(np.float32)
    img = (img - mean) / std
    tensor = torch.from_numpy(img)
    return tensor


def label_to_tensor(label, threshold=0.5):
    label = (label > threshold).astype(np.float32)
    tensor = torch.from_numpy(label).type(torch.FloatTensor)
    return tensor


def tensor_to_image(tensor, mean=0, std=1):
    image = tensor.numpy()
    image = image * std + mean
    image = image * 255
    image = image.astype(dtype=np.uint8)
    return image


def tensor_to_label(tensor):
    label = tensor.numpy() * 255
    label = label.astype(dtype=np.uint8)
    return label


def load_names(path, postfix):
    path = join(path, 'data')
    #print('path', path)
    full_names = glob(path + os.sep +'*.' + postfix)
    #print('full_names', full_names)
    names = [name.split('.')[0].split(os.sep)[-1] for name in full_names]#'/'
    return names

class KidneyData(Dataset):
    def __init__(self, path, mode='train'):  #transforms, mask_ration=0.2
        super(KidneyData, self).__init__()
        self._path = path
        #self._transforms = transforms
        #self._mask_ratio = mask_ration
        self._mode = mode
        self._names = load_names(path, 'png')

    def get_img(self, index):
        name = self._names[index]
        #print('imgname', name)
        img = imread(join(self._path, 'data') + os.sep + name + '.png')
        return img

    def get_label(self, index):
        name = self._names[index]
        #print('labelname', name)
        label = imread(join(self._path, 'bon') + os.sep + name + '.png')
        return label

    def get_train_item(self, index):
        #print('get train_item')
        img = self.get_img(index).astype(np.float32)
        label = self.get_label(index)
        #label = label.astype(np.float32)
        #img += label * self._mask_ratio
        '''
        for t in self._transforms:
            img, label = t(img, label)
        '''
        intensity_min = np.min(img)
        intensity_max = np.max(img)
        img = (img - intensity_min) / (intensity_max - intensity_min)
        label = label.astype(np.float32) / 255
        img = img_to_tensor(img)
        label = label_to_tensor(label)
        return img, label, index

    def get_test_item(self, index):
        img = self.get_img(index)
        img = img_to_tensor(img)
        return img, index

    def __getitem__(self, index):
        if self._mode == 'train':
            #print('get train_item')
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    def __len__(self):
        return len(self._names)

    '''
def check_dataset():
    def train_augment(img, label):
        img, label = resizeN([img, label], (512, 512)) #(256, 256))  #(1024, 1024))  #
        #img, label = random_horizontal_flipN([img, label])
        return img, label

    def test_augment(img, label):
        img, label = resizeN([img, label], (512, 512)) #(256, 256))  #(1024, 1024)
        return img, label

    test_data = KidneyData('.test/', transforms=[lambda x, y: train_augment(x, y), ], mode='train')
    loader = DataLoader(test_data, batch_size=1, num_workers=1)

    for it, (img, label, indices) in enumerate(loader):
        img = tensor_to_image(img, mean=0, std=1)
        label = tensor_to_label(label)
        imsave('img.png', img.reshape((512, 512)))  #(1024, 1024)
        imsave('label.png', label.reshape((512, 512)))  #(1024, 1024)


if __name__=='__main__':
    check_dataset()
'''