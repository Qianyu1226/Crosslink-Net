import os
import torch
from dataset.BSDDataLoader import get_training_set,get_test_set
from modules.CrossWnet import UNet
from glob import glob
from torch.utils.data import DataLoader
from dataset.transforms import *
from tqdm import tqdm
import copy
from dataset.KidneySet import img_to_tensor, label_to_tensor
from torch.autograd import Variable
from skimage.io import imsave, imread
from skimage.transform import resize
from utils.visualize import visulize_gt
#import torch.nn.functional as F
from utils.statistic import dice_loss
from os.path import exists, join, abspath

test_path=join(abspath('.'),'COVID-19','test','data')
label_path=join(abspath('.'),'COVID-19','test','bon')

out_dir = './out/result/'
#checkpoint = 10
resized_shape = (512, 512)
batch_size = 1
in_shape = (1, 512, 512)
is_bn = True
use_cuda=False
device = torch.device("cuda: 2" if use_cuda else "cpu")
target_mode='bon'
colordim=1
resizenum=512

os.makedirs(out_dir + '/epoch12', exist_ok=True)
test_dataset = get_test_set(resizenum, target_mode=target_mode, colordim=colordim)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)
# network setting
net = UNet(in_shape, is_bn)
model_file = './checkpoint/'+ 'model_epoch_12' + '.pkl'
print(model_file)
net.load_state_dict(torch.load(model_file))
net.to(device)
net.eval()

names =glob(test_path + '/*.png') #jpg
names = [name.split('.')[0].split('/')[-1] for name in names] #\\ for windows, / for linux

dices = {}
def val_augment(img, label):
    img, label = resizeN([img, label], resized_shape)
    return img, label

# start prediction
for name in tqdm(names):
    print(name)
    img = imread(test_path+'/' + name + '.png')  #jpg
    tmp_img = copy.copy(img)
    gt_label = imread(label_path +'/'+ name + '.png')
    row, col = np.shape(img)
    img, label = val_augment(img, gt_label)

    img = img.astype(np.float32) / 255
    label = label.astype(np.float32) / 255
    img = img.reshape((1,512,512))#296,296
    label = label.reshape((1,512,512))#296,296
    img = img_to_tensor(img)
    label = label_to_tensor(label)
    img = Variable(img.to(device), volatile=True)
    label = Variable(label.to(device), volatile=True)

    logits, v_atten, h_atten = net(img)  #, h_fmaps
    probs = torch.sigmoid(logits)
    masks = (probs > 0.5).float()
    dice_acc = dice_loss(masks, label)
    dices[name] = dice_acc.item()
    masks = masks.data.cpu().numpy()
    masks = masks.squeeze()
    masks = resize(masks, (row, col), mode='reflect', preserve_range=True)  #(296, 296)
    imsave(out_dir + '/epoch12/' + name + '.bmp', masks.astype(np.uint8)*255) #
    result_vis = visulize_gt(tmp_img, gt_label, 'r')
    result_vis = visulize_gt(result_vis, masks.astype(np.uint8)*255, 'g')
    imsave(out_dir + '/epoch12/' + name + '_1.png', result_vis.astype(np.uint8))

total = 0
for key, value in dices.items():
     print(key + '   ' + str(value))
     total += value
print(total / len(dices))
