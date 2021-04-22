# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:45:19 2021

@author: Jack
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, random_split
from ReNetutility import Conv_Cat_Residual3
from sklearn.metrics import confusion_matrix
from scipy.io import savemat
import time
from torchsummary import summary

mat = scipy.io.loadmat('/home/zhuochengjiang/project-seong/Bearing_FFT_Data_Restruct.mat')
#mat = scipy.io.loadmat('/home/zhuochengjiang/project-seong/Bearing_FFT_Data_Recon_Sub2.mat')

torch.manual_seed(2)
y_pred = []
y_true = []
# Model initialization
def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('linear') != -1:
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)



def generate_data(mat, b_size):
    AE = mat['yffAE']
    VB = mat['yffVB']
    classes = mat['runtime']
    
    AE = np.array(AE)
    VB = np.array(VB)
    classes = np.array(classes)
    
    AE = torch.from_numpy(AE).type(torch.FloatTensor)
    VB = torch.from_numpy(VB).type(torch.FloatTensor)
    classes = torch.from_numpy(classes).type(torch.FloatTensor)
    
    AE = AE.view(len(AE), 1, -1)
    VB = VB.view(len(VB), 1, -1)
    data = torch.stack([AE, VB], 0)
    data = data.permute(1,0,2,3)
    data = torch.squeeze(data)
    '''
    print(data.size())
    data = norm(data)
    print(data.mean())
    print(data.std())
    '''
    classes = classes.view(len(classes), 1)
    
    train_size = int(0.8*len(data))
    testval_size = len(AE) - train_size
    val_size = int(testval_size*0.5)
    test_size = testval_size - val_size
    
    dataset_AE = TensorDataset(data, classes)
    trainAE_set, testvalAE_set = random_split(dataset_AE, [train_size, testval_size])
    testAE_set, valAE_set = random_split(testvalAE_set, [test_size, val_size])
    
    loader_trainAE = DataLoader(trainAE_set, batch_size=b_size, 
                    num_workers=0, shuffle=True)
    loader_testAE = DataLoader(testAE_set, batch_size=b_size, 
                    num_workers=0, shuffle=True)
    loader_valAE = DataLoader(valAE_set, batch_size=b_size, 
                    num_workers=0, shuffle=True)
    
    return loader_trainAE, train_size, loader_testAE, test_size, loader_valAE, val_size 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

batch_size = 64
num_epochs = 200

loader_trainAE, num_trainAE, loader_testAE, num_testAE, loader_valAE, num_valAE = generate_data(mat, batch_size)



resnet = Conv_Cat_Residual3(input_channel = 1, layers = [1,1,1], regression_out = 1)
count = count_parameters(resnet)
#summary(resnet, [1, 5000], [1, 5000])

print(count)
resnet.apply(init_weight)
resnet = resnet.cuda()
criterion = nn.MSELoss().cuda()

optimizer = torch.optim.Adam(resnet.parameters(), lr = 0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1)

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    resnet.train()
    scheduler.step()
    for i, (samples, labels) in enumerate(loader_trainAE):
        sample_AE = torch.as_tensor(samples[:, 0:1, :])
        sample_VB = torch.as_tensor(samples[:, 1:2, :])
        samplesAE = Variable(sample_AE.cuda())
        samplesVB = Variable(sample_VB.cuda())
        labels = labels.squeeze()
        labelsV = Variable(labels.cuda())
        
        optimizer.zero_grad()
        predict_label = resnet(samplesAE, samplesVB)
        predict_label = predict_label.reshape(-1)
        #print('Predict:', predict_label)
        #print('labels:', labelsV)
        loss = torch.sqrt(criterion(predict_label, labelsV))

        loss.backward()
        optimizer.step()
    
    resnet.eval()
    loss_x = 0
    num_batchs = 0
    for i, (samples, labels) in enumerate(loader_trainAE):
        num_batchs = i
        with torch.no_grad():
            sample_AE = torch.as_tensor(samples[:, 0:1, :])
            sample_VB = torch.as_tensor(samples[:, 1:2, :])
            samplesAE = Variable(sample_AE.cuda())
            samplesVB = Variable(sample_VB.cuda())
            labels = labels.squeeze()
            labelsV = Variable(labels.cuda())

            predict_label = resnet(samplesAE, samplesVB)
            predict_label = predict_label.reshape(-1)
            #print(len(predict_label))
            #print(labelsV)
            loss = torch.sqrt(criterion(predict_label, labelsV))
            loss_x += loss.item()

    print("Training average RMSE:", float(loss_x/(num_batchs+1)))


    loss_x = 0
    num_batchs = 0
    for i, (samples, labels) in enumerate(loader_valAE):
        num_batchs = i
        with torch.no_grad():
            sample_AE = torch.as_tensor(samples[:, 0:1, :])
            sample_VB = torch.as_tensor(samples[:, 1:2, :])
            samplesAE = Variable(sample_AE.cuda())
            samplesVB = Variable(sample_VB.cuda())
            labels = labels.squeeze()
            labelsV = Variable(labels.cuda())

            predict_label = resnet(samplesAE, samplesVB)
            predict_label = predict_label.reshape(-1)
            #print(len(predict_label))
            #print(labelsV)
            loss = torch.sqrt(criterion(predict_label, labelsV))
            loss_x += loss.item()

    print("Validation average RMSE:", float(loss_x/(num_batchs+1)))


    loss_x = 0
    mean_loss = 0
    num_batchs = 0
    for i, (samples, labels) in enumerate(loader_testAE):
        num_batchs = i
        with torch.no_grad():
            sample_AE = torch.as_tensor(samples[:, 0:1, :])
            sample_VB = torch.as_tensor(samples[:, 1:2, :])
            samplesAE = Variable(sample_AE.cuda())
            samplesVB = Variable(sample_VB.cuda())
            labels = labels.squeeze()
            labelsV = Variable(labels.cuda())

            predict_label = resnet(samplesAE, samplesVB)
            predict_label = predict_label.reshape(-1)
            if epoch == num_epochs - 1:
                for i in range(len(predict_label)):
                    y_pred.append(float(predict_label[i].cpu()))
                    y_true.append(float(labelsV[i].cpu()))
            for j in range(len(predict_label)):
                mean_loss += float(torch.abs(predict_label[j] - labelsV[j]).cpu())
            
            loss = torch.sqrt(criterion(predict_label, labelsV))
            loss_x += loss.item()
        
    print("Test RMSE:", float(loss_x / (num_batchs+1)))
    print("Test abs error:", float(mean_loss / num_testAE))

savemat("/home/zhuochengjiang/project-seong/paper_AEVB_50000.mat", {'Pred':y_pred, 'True':y_true})
