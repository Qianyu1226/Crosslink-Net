# coding:utf-8
import torch.nn as nn
import torch
import numpy as np
#import torch.nn.functional as F

class CovLoss(nn.Module):
    def __init__(self):
        super(CovLoss, self).__init__()
        self._bce_loss = nn.BCELoss()  # nn.CrossEntropyLoss(reduce=True, size_average=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, logits, labels, v_attention, h_attention):  #
          # BCEloss
        logits_flatten = logits.view(-1)
        labels_flatten = labels.view(-1)
        bceloss = self._bce_loss(logits_flatten, labels_flatten)

        # Diceloss
        N = labels.size(0)
        smooth = 1
        masks = (logits > 0.4).float()
        input_flat = masks.view(N, -1)
        target_flat = labels.view(N, -1)
        intersection = input_flat * target_flat
        # 
        diceloss = torch.tensor(2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth),
                                requires_grad=True)
        #print('diceloss is a leaf: ', diceloss.is_leaf)
        diceloss = 1 - diceloss.sum() / N
        #print('2nd diceloss is a leaf: ', diceloss.is_leaf)
        #######################################################
        #Covloss
        #设v_attention.size=[4, 64, 64]

        N1 = v_attention.size(0)
        N2 = v_attention.size(2)
        #N3 = v_features.size(1)
        #print("before poooling ")
        attent_labels = self.avgpool(self.avgpool(labels)) #block3, pooling twice， the same size with v_flatten
        v_flatten = v_attention.view(N1, N2 * N2)
        h_flatten = h_attention.view(N1, N2 * N2)
        l_flatten = attent_labels.view(N1, N2 * N2)
        vhl = torch.randn(N1, N2*N2, 3)#, 3
        for i in range(N1):
            vhl[i,:,:] = torch.cat((v_flatten[i,:].unsqueeze(1),h_flatten[i,:].unsqueeze(1),l_flatten[i,:].unsqueeze(1)),1)
            #vhl[i, :, :] = torch.cat((v_flatten[i, :].unsqueeze(1), h_flatten[i, :].unsqueeze(1)), 1)
        vhl_mean = torch.randn(N1, 3)
        for i in range(N1):
            for j in range(3):
                vhl_mean[i, j] = vhl[i, :, j].mean()
        vhl_cor = torch.randn(N1)
        #vhl_cor_numpy = vhl_cor.detach().numpy()
        for i in range(N1):
            vhl_cor[i] = ((vhl[i, :, 0] - vhl_mean[i, 0])*(vhl[i, :, 1] - vhl_mean[i, 1])*(vhl[i, :, 2] - vhl_mean[i, 2])).sum(0)\
                         /torch.sqrt((vhl[i, :, 0] - vhl_mean[i, 0]).pow(2).sum(0)*(vhl[i, :, 1] - vhl_mean[i, 1]).pow(2).sum(0)*(vhl[i, :, 2] - vhl_mean[i, 2]).pow(2).sum(0))

        #vhl_cor = torch.from_numpy(vhl_cor_numpy)
        #print('vhl_cor: ', vhl_cor.is_leaf)
        cor_loss = torch.tensor(-vhl_cor.sum()/N1).cuda()  ##如果在GPU上跑要带上cuda
        #print('cor_loss is a leaf: ',cor_loss.is_leaf)
        #
        return 0.2*bceloss + 0.3*diceloss + 0.5*cor_loss
        #return bceloss*0.5 + diceloss*0.5 #