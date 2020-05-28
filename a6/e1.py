import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

import torchvision
import torchvision.transforms as transforms
import cv2
import sys
import numpy
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from simpleNet import Net

PATH = 'networks/e1_net.pth'
OUT = 'networks/'

random_seed = 1
torch.manual_seed(random_seed)

def trainNetwork(network,trainLoader,number_of_epochs= 3,optimizer=None):
    start_time = time.time()

    if not optimizer:
        optimizer =optim.Adam(network.parameters())

    print("start training")
    criterion = nn.CrossEntropyLoss() #

    network.train()

    for epoch in range(number_of_epochs):
        trainEpoch(network,trainLoader,criterion,optimizer,epoch)
                
    torch.save(network.state_dict(), PATH) #save after training network
    print("finish training")
    print("Time : ",time.time()-start_time," sec")

def trainEpoch(network,trainLoader,criterion,optimizer,epoch=1):
    running_loss = 0.0
    for i,data in enumerate(trainLoader , 0):
        image, labels = data

        optimizer.zero_grad()
        outputs = network(image)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # TODO: better print
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            
            #checkpoint save
            torch.save(network.state_dict(), OUT + 'model.pth') 
            torch.save(optimizer.state_dict(), OUT + 'optimizer.pth')

def testNetwork(network,testLoader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            image, labels = data
            outputs = network(image)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network: %.3f %%' % (
    100 * correct / total))


def loadSets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # the datasets
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) # download Trainingset
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform) # download Testset

    return trainset, testset

def create_subset(dataset,minval,maxval):
    idx = (dataset.targets <= maxval) & (dataset.targets >= minval)
    dataset.targets = dataset.targets[idx]
    dataset.data = dataset.data[idx]
    return dataset

def changePath(path):
    global PATH
    PATH = path

if __name__ == "__main__":

    # the datasets
    trainset, testset = loadSets()

    trainset = create_subset(trainset,0,4)
    testset = create_subset(testset,0,4)

    trainLoader = torch.utils.data.DataLoader(trainset, shuffle=True,batch_size=100)
    testLoader = torch.utils.data.DataLoader(testset, shuffle=True,batch_size=100)

    network = Net()
    print(network)

    if len(sys.argv) == 2 and sys.argv[1] == "trainmode":
        trainNetwork(network,trainLoader,number_of_epochs = 3)
    else:
        try:
            network.load_state_dict(torch.load(PATH)) #load last training
        except:
            trainNetwork(network,trainLoader,number_of_epochs = 3)
    
    testNetwork(network,testLoader)
