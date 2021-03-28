# coding:utf-8
import os
import torch
from modules.CrossWnet import UNet
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.transforms import *
from torch.autograd import Variable
from dataset.BSDDataLoader import get_training_set,get_test_set
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from modules.AttentionLoss import CovLoss  #for Net3, Net_mul_atten, Net_mul_pool
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"  #
from skimage.io import imsave, imread
"""Global Setting"""

out_dir = './out/'
initial_checkpoint = None #'./checkpoint/'+ 'model_epoch_19' + '.pkl'     #First train is NONE

# resized_shape = (256, 256)
in_shape = (1, 512, 512)#192
batch_size =2 #4
is_bn = True
loss_layer=3  #the block for loss
featuremaps=128  #featuremaps on the block
dime=128  

num_epoches = 15
it_smooth = 20
use_cuda= True #False#GPU
device = torch.device("cuda: 2" if use_cuda else "cpu")


os.makedirs(out_dir, exist_ok=True)

target_mode='bon'
colordim=1
resizeshape= 512 #1024 512 192
train_dataset = get_training_set(resizeshape, target_mode=target_mode, colordim=colordim)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)

# Random Setting
SEED = 235202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# network setting
net = UNet(in_shape, is_bn).to(device)
# optimizer settings
optimizer = Adam(lr=5e-6, params=net.parameters(), weight_decay=0.005) #5e-6

epoch_valid = list(range(0, num_epoches + 1))
epoch_save = list(range(0, num_epoches + 1))

start_epoch = 0 #if training not from start, then it is the num of model_epoch_X.pkl
#net.load_state_dict(torch.load(initial_checkpoint)) # training from model_epoch_X.pkl
my_loss = CovLoss().to(device)
losses_history=[]
for epoch in range(start_epoch, num_epoches + 1):
    print(epoch)
    if epoch == num_epoches:
        break
    epoch_loss = 0
    # start to train current epoch
    net.train()
    #print('start train %02d' % epoch)
    for it, (images, labels) in enumerate(train_loader):  #, indices
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        #print("it:", it)
        #print("sample_fname: ", sample_fname)
        # forward
        logits, v_featuremap, h_featuremap = net(images)  #v_featuremap, h_featuremap for Net2,mul_mean; logits, v_atten for Net3 4,atten
        #print("v_f type: ", type(v_featuremap))
        probs = torch.sigmoid(logits)
        
        loss = my_loss(probs, labels, v_featuremap, h_featuremap) #
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #
        if it % 5 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, it, len(train_loader), loss.item()))
       #the comment here
    losses_history.append(epoch_loss/ len(train_loader))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_loader)))	
    #if epoch % 2 is 0: #
    model_out_path = "./checkpoint/model_epoch_{}.pkl".format(epoch)
    torch.save(net.state_dict(), model_out_path)
# save last
#torch.save(net.state_dict(), out_dir + '/snap/final.pth')
