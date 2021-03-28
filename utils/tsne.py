from __future__ import print_function

import argparse
import csv
import os
import os.path
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
batch_size = 4

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# out_dir = '/data/jinquan/data/cvpr_kidney/out/unet_10_26/'
# test_path = '/data/jinquan/data/cvpr_kidney/input/kidney/18_3_4_test/'

out_dir = '/data/jinquan/data/cvpr_kidney/out/unet_18_3_6/'
test_path = '/data/jinquan/data/cvpr_kidney/input/kidney/18_3_6/test/'

checkpoint = 4
resized_shape = (256, 256)
batch_size = 1
in_shape = (1, 256, 256)
is_bn = False


def inference():
    os.makedirs(out_dir + '/whole_seg', exist_ok=True)

    def test_augment(img, label):
        img, label = resizeN([img, label], resized_shape)
        return img, label

    test_dataset = KidneyData(test_path, transforms=[lambda x, y: test_augment(x, y), ], mode='train')
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False,
                            num_workers=4, pin_memory=False)

    # network setting
    net = UNet(in_shape, is_bn)
    model_file = out_dir + 'snap/final' + '.pth'
    print(model_file)
    net.load_state_dict(torch.load(model_file))
    net.cuda()
    net.eval()

    names =glob(test_path + '*.jpg')
    names = [name.split('.')[0].split('/')[-1] for name in names]

    dices = {}

    def val_augment(img, label):
        img, label = resizeN([img, label], resized_shape)
        return img, label

    # start prediction
    for name in tqdm(names):
        img = imread(test_path + name + '.jpg')
        gt_label = imread(test_path + name + '_1.bmp')

        img, label = val_augment(img, gt_label)

        img = img.astype(np.float32) / 255
        label = label.astype(np.float32) / 255
        img = img.reshape((1,256,256))
        label = label.reshape((1,256,256))
        img = img_to_tensor(img)
        label = label_to_tensor(label)
        img = Variable(img.cuda(), volatile=True)
        label = Variable(label.cuda(), volatile=True)

        logits = net(img)
        probs = F.sigmoid(logits)
        masks = (probs > 0.5).float()
        dice_acc = dice_loss(masks, label)
        dices[name] = dice_acc.data[0]
        masks = masks.data.cpu().numpy()
        masks = masks.squeeze()
        masks = resize(masks, (296, 296), mode='reflect', preserve_range=True)
        imsave(out_dir + 'whole_seg/' + name + '.bmp', masks)
        result_img = visulize_gt(masks.astype(np.uint8)*255, gt_label)
        imsave(out_dir + 'whole_seg/' + name + '_1.bmp', result_img.astype(np.uint8))

    total = 0
    for key, value in dices.items():
        print(key + '   ' + str(value))
        total += value
    print(total / len(dices))


def tsne(model_file, test_data_dir, out_dir):
    if model_file is None:
        model = models.__dict__['resnet50'](pretrained=True)
        model.fc = nn.Linear(2048, 2)
    else:
        model = models.__dict__['resnet50']()
        model.fc = nn.Linear(2048, 2)
        model.load_state_dict(torch.load(model_file))

    new_model = nn.Sequential(*list(model.children())[:-1])
    new_model.cuda()
    new_model.eval()

    normalize = transforms.Normalize(mean=[0.3, 0.6, 0.4], std=[0.129, 0.1, 0.125])

    test_loader = data.DataLoader(
        datasets.ImageFolder(test_data_dir,
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    vectors = []
    labels = []
    for i, (images, target) in enumerate(test_loader):
        # measure data loading time

        target = target.cpu().numpy()
        image_var = torch.autograd.Variable(images).cuda()

        vector = new_model(image_var)
        vector = vector.data.cpu().numpy()
        vector = np.reshape(vector, (batch_size, -1))
        vectors.append(vector)
        labels.append(target)

    vectors = np.asarray(vectors)
    vectors = np.reshape(vectors, (-1, 2048))
    labels = np.asarray(labels)
    labels = labels.flatten()

    print('start tsne')
    t0 = time.time()
    tsne = TSNE(perplexity=30, n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(vectors)
    t1 = time.time()
    print(t1-t0)
    plt.figure()
    # colors = ['r']*2000
    # colors[:1000] = ['y']*1000
    plt.scatter(Y[:,0]/8, Y[:,1]/8, c = labels, cmap=plt.cm.Spectral)
    # plt.show()
    plt.savefig('vis_superior.eps')


if __name__ == '__main__':
    inference(None, '/data/jinquan/data/dog_vs_cat_data/val/', '/')
    # inference('./snap/056.pth', '/data/jinquan/data/dog_vs_cat_data/val/', '/')
