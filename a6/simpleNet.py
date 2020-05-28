import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self,input_dim = 784,hidden_dim =100 ,output_dim =5,number_of_hidden_layers = 5):
        super(Net, self).__init__()

        self.number_of_hidden_layers = number_of_hidden_layers

        self.fc1 = nn.Linear(input_dim, hidden_dim)  # input layer 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # hidden layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim) # hidden layer
        self.fc4 = nn.Linear(hidden_dim, hidden_dim) # hidden layer
        self.fc5 = nn.Linear(hidden_dim, hidden_dim) # hidden layer
        self.fc6 = nn.Linear(hidden_dim, hidden_dim) # hidden layer
        self.fc7 = nn.Linear(hidden_dim, output_dim) # output layer

        self.dropout = nn.Dropout()
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        
        #xavier init
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        torch.nn.init.xavier_uniform_(self.fc6.weight)
        torch.nn.init.xavier_uniform_(self.fc7.weight)

    def forward(self, x):

        x = x.view(-1, self.num_flat_features(x)) #2d img (tensor) to 1d tensor

        x = F.relu(self.bn1(self.fc1(x))) #self.dropout(self.bn1(...)) macht das Sach schlimmer
        x = F.relu(self.bn1(self.fc2(x))) # self.dropout(self.bn1(...)))
        x = F.relu(self.bn1(self.fc3(x)))
        x = F.relu(self.bn1(self.fc4(x)))
        x = F.relu(self.bn1(self.fc5(x)))
        x = F.relu(self.bn1(self.fc6(x)))

        x = self.fc7(x)
        return F.log_softmax(x,1)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features