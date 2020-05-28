import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

import torchvision
import torchvision.transforms as transforms
import cv2
import sys
import e1

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from simpleNet import Net

def limit_set(dataset,limit,valrange):
    targetList = []
    dataList = []

    for i in range(valrange[0], valrange[1]):
        idx = dataset.targets == i
        
        targetList.append(dataset.targets[idx][:limit])
        dataList.append(dataset.data[idx][:limit])
    
    dataset.targets = torch.cat(tuple(targetList))
    dataset.data = torch.cat(tuple(dataList))
    dataset.targets -= valrange[0]
    return dataset

def freez_all_hidden(network):
    network.fc2.weight.requires_grad = False
    network.fc2.bias.requires_grad = False

    network.fc3.weight.requires_grad = False
    network.fc3.bias.requires_grad = False

    network.fc4.weight.requires_grad = False
    network.fc4.bias.requires_grad = False

    network.fc5.weight.requires_grad = False
    network.fc5.bias.requires_grad = False

    network.fc6.weight.requires_grad = False
    network.fc6.bias.requires_grad = False

def freez_last_3_hidden(network):
    network.fc4.weight.requires_grad = False
    network.fc4.bias.requires_grad = False

    network.fc5.weight.requires_grad = False
    network.fc5.bias.requires_grad = False

    network.fc6.weight.requires_grad = False
    network.fc6.bias.requires_grad = False

if __name__ == "__main__":

    trainset, testset = e1.loadSets()

    trainset = limit_set(trainset,100,(5,10)) # only the first 100 values from one label
    testset = limit_set(testset,100,(5,10)) # only the first 100 values from one label

    trainLoader = torch.utils.data.DataLoader(trainset, shuffle=True,batch_size=100)
    testLoader = torch.utils.data.DataLoader(testset, shuffle=True,batch_size=100)

    e1.changePath("networks/e2_net.pth")

    network = Net()
    print(network)

    if len(sys.argv) == 2:
        if sys.argv[1] == "trainmode":
            e1.trainNetwork(network,trainLoader,number_of_epochs = 3)
        elif sys.argv[1] == "3.":
            network.load_state_dict(torch.load(e1.PATH))
            
            #reinit one hidden layer
            network.fc2 = nn.Linear(100,100)
            torch.nn.init.xavier_uniform_(network.fc2.weight)

            e1.trainNetwork(network,trainLoader,number_of_epochs = 3)
        elif sys.argv[1] == "4.":
            network.load_state_dict(torch.load(e1.PATH))
            freez_last_3_hidden(network)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, network.parameters()))

            e1.trainNetwork(network,trainLoader,number_of_epochs = 3,optimizer= optimizer)
    else:
        try:
            network.load_state_dict(torch.load(e1.PATH)) #load last training
            freez_all_hidden(network)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, network.parameters()))

            e1.trainNetwork(network,trainLoader,number_of_epochs = 3,optimizer= optimizer)
        except:
            e1.trainNetwork(network,trainLoader,number_of_epochs = 3)
    
    e1.testNetwork(network,testLoader)

