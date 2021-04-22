# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:45:19 2020

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
from ReNetutility import MyReNet1D
from sklearn.metrics import confusion_matrix
import time


#mat = scipy.io.loadmat('/home/zhuochengjiang/project-seong/Bearing_FFT_Data_Restruct.mat')
mat = scipy.io.loadmat('/home/zhuochengjiang/project-seong/Bearing_FFT_Data_Recon_Sub2.mat')
y_pred = []
y_true = []
# Model initialization

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('linear') != -1:
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)
        
# Data Preparation
def generate_AE(mat, b_size):
    
    AE = mat['yffAE2']
    classes = mat['runtime_class']
    for i in range(len(classes)):
        classes[i] = classes[i]-1
    AE = np.array(AE)
    classes = np.array(classes)
    AE = torch.from_numpy(AE).type(torch.FloatTensor)
    classes = torch.from_numpy(classes).type(torch.LongTensor)
    AE = AE.view(len(AE), 1, -1)
    classes = classes.view(len(classes), 1)
    
    train_size = int(0.8*len(AE))
    testval_size = len(AE) - train_size
    val_size = int(testval_size*0.5)
    test_size = testval_size - val_size
    
    dataset_AE = TensorDataset(AE, classes)
    trainAE_set, testvalAE_set = random_split(dataset_AE, [train_size, testval_size])
    testAE_set, valAE_set = random_split(testvalAE_set, [test_size, val_size])
    
    loader_trainAE = DataLoader(trainAE_set, batch_size=b_size, 
                    num_workers=0, shuffle=True)
    loader_testAE = DataLoader(testAE_set, batch_size=b_size, 
                    num_workers=0, shuffle=True)
    loader_valAE = DataLoader(valAE_set, batch_size=b_size, 
                    num_workers=0, shuffle=True)
    
    return loader_trainAE, train_size, loader_testAE, test_size, loader_valAE, val_size 

def generate_VB(mat, b_size):
    
    VB = mat['yffVB2']
    classes = mat['runtime_class']
    for i in range(len(classes)):
        classes[i] = classes[i]-1
    VB = np.array(VB)
    classes = np.array(classes)
    
    VB = torch.from_numpy(VB).type(torch.FloatTensor)
    classes = torch.from_numpy(classes).type(torch.LongTensor)
    VB = VB.view(len(VB), 1, -1)
    classes = classes.view(len(classes), 1)
    
    train_size = int(0.8*len(VB))
    testval_size = len(VB) - train_size
    val_size = int(testval_size*0.5)
    test_size = testval_size - val_size
    
    dataset_VB = TensorDataset(VB, classes)
    trainVB_set, testvalVB_set = random_split(dataset_VB, [train_size, testval_size])
    testVB_set, valVB_set = random_split(testvalVB_set, [test_size, val_size])
    
    loader_trainVB = DataLoader(trainVB_set, batch_size=b_size, 
                    num_workers=0, shuffle=True)
    loader_testVB = DataLoader(testVB_set, batch_size=b_size, 
                    num_workers=0, shuffle=True)
    loader_valVB = DataLoader(valVB_set, batch_size=b_size, 
                    num_workers=0, shuffle=True)
    
    return loader_trainVB, train_size, loader_testVB, test_size, loader_valVB, val_size 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

batch_size = 64
num_epochs = 200

loader_trainAE, num_trainAE, loader_testAE, num_testAE, loader_valAE, num_valAE = generate_VB(mat, batch_size)


resnet = MyReNet1D(input_channel = 1, layers = [1,1,1], num_classes = 4)
count = count_parameters(resnet)
print(count)
resnet.apply(init_weight)
resnet = resnet.cuda()

criterion = nn.CrossEntropyLoss(size_average=False).cuda()

optimizer = torch.optim.Adam(resnet.parameters(), lr = 0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1)
train_loss = np.zeros([num_epochs, 1])
test_loss = np.zeros([num_epochs, 1])
train_acc = np.zeros([num_epochs, 1])
test_acc = np.zeros([num_epochs, 1])
start_time = time.time()

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    resnet.train()
    scheduler.step()
    loss_x = 0
    for i, (samples, labels) in enumerate(loader_trainAE):
        samplesV = Variable(samples.cuda())
        labels = labels.squeeze()
        labelsV = Variable(labels.cuda())

        optimizer.zero_grad()
        predict_label = resnet(samplesV)
        loss = criterion(predict_label, labelsV)
        #print(loss)
        loss_x += loss.item()

        loss.backward()
        optimizer.step()
        
        
    train_loss[epoch] = loss_x / num_trainAE
    print("Training loss:", (100*train_loss[epoch]))
    
    resnet.eval()
    # loss_x = 0
    correct_train = 0
    for i, (samples, labels) in enumerate(loader_trainAE):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labels = labels.squeeze()
            labelsV = Variable(labels.cuda())

            predict_label = resnet(samplesV)
            prediction = predict_label.data.max(1)[1]
            correct_train += prediction.eq(labelsV.data.long()).sum() 
            # loss = criterion(predict_label, labelsV)
            # loss_x += loss.item()

    print("Training accuracy:", (100*float(correct_train)/num_trainAE))
    # train_acc[epoch] = 100*float(correct_train)/num_trainAE
    
    correct_val = 0
    for i, (samples, labels) in enumerate(loader_valAE):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labels = labels.squeeze()
            labelsV = Variable(labels.cuda())

            predict_label = resnet(samplesV)
            prediction = predict_label.data.max(1)[1]
            correct_val += prediction.eq(labelsV.data.long()).sum() 
            # loss = criterion(predict_label, labelsV)
            # loss_x += loss.item()

    print("Validation accuracy:", (100*float(correct_val)/num_valAE))

    correct_test = 0
    for i, (samples, labels) in enumerate(loader_testAE):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labels = labels.squeeze()
            labelsV = Variable(labels.cuda())
            # labelsV = labelsV.view(-1)

            predict_label = resnet(samplesV)
            prediction = predict_label.data.max(1)[1]
            if epoch == num_epochs-1:
                for i in range(len(prediction)):
                    y_pred.append(int(prediction[i].cpu()))
                    y_true.append(int(labelsV[i].cpu()))
            correct_test += prediction.eq(labelsV.data.long()).sum()

            # loss = criterion(predict_label, labelsV)
            #loss_x += loss.item()

    print("Test accuracy:", (100 * float(correct_test) / num_testAE))
    
    '''
    test_loss[epoch] = loss_x / num_testAE
    test_acc[epoch] = 100 * float(correct_test) / num_testAE
    testacc = str(100 * float(correct_test) / num_testAE)[0:6]
    '''

print("---%s seconds ---" % (time.time() - start_time))
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)
#torch.save(resnet.state_dict(), '/home/zhuochengjiang/project-seong/reset-AE5000-classification.pt')


