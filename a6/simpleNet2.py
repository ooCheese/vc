import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self,input_dim = 784,hidden_dim =100 ,output_dim =5,number_of_hidden_layers = 5):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 3)
        #self.conv2 = nn.Conv2d(6, 16, 3)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # input layer 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim) # output layer

        self.dropout = nn.Dropout()

        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)

        self.number_of_hidden_layers = number_of_hidden_layers
        
        #xavier init
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):

        x = x.view(-1, self.num_flat_features(x)) #2d img (tensor) to 1d tensor
        x = F.relu(self.bn1(self.fc1(x))) #self.dropout(self.bn1(...)) macht das Sach schlimmer
        for i in range(self.number_of_hidden_layers):
            x = F.relu(self.bn1(self.fc2(x))) # self.dropout(self.bn1(...))) 
        x = self.fc3(x)
        return F.log_softmax(x,1)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features