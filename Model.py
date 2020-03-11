import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

# Credict: https://github.com/kefirski/pytorch_Highway
class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear
        return x
    
class Model(nn.Module):
    
    def __init__(self, weeks, time_slots, features, drop_out, device=None):
        
        super(Model, self).__init__()
        self.device = device
        self.weeks = weeks
        self.time_slots = time_slots
        
        self.time_slot_GRU_1 = nn.GRU(features, 256, 1, batch_first=True, bidirectional=True)
        self.time_slot_Linear_1 = nn.Linear(256*2, 256)
        self.norm_1 = torch.nn.BatchNorm1d(time_slots)
        
        self.time_slot_Linear_3 = nn.Linear(256, 256)
        self.time_slot_MaxPool = nn.MaxPool1d(256)
        
        self.last_two_GRU_1 = nn.GRU(time_slots, 256, 1, batch_first=True, bidirectional=True)
        self.last_two_Linear_1 = nn.Linear(256*2, 256)

        
        self.week_GRU_1 = nn.GRU(time_slots, 256, 1, batch_first=True, bidirectional=True)
        self.week_linear_1 = nn.Linear(256*2, 256)
        self.norm_3 = torch.nn.BatchNorm1d(weeks)

        self.week_GRU_2 = nn.GRU(256, 256, 1, batch_first=True, bidirectional=True)
        self.MaxPool = nn.MaxPool1d(256*2)
        self.norm_4 = torch.nn.BatchNorm1d(34)
        
        self.highway_1 = Highway(34, 3, f=torch.nn.functional.relu)
        self.linear_5 = torch.nn.Linear(34, 28)        
        
        self.drop = nn.Dropout(drop_out)
        
    def forward(self, x):
        x = x.permute(1, 0, 2, 3) 
        collect = torch.Tensor().type(torch.FloatTensor).to(self.device)
        for idx in x:
            layer_1, _ = self.time_slot_GRU_1(idx)
            layer_1 = self.time_slot_Linear_1(self.drop(layer_1.contiguous().view(-1, layer_1.size(2))))
            layer_1 = F.selu(self.norm_1(layer_1.view(-1, idx.size(1), 256)))
            
            layer_2 = self.time_slot_Linear_3(self.drop(layer_1))
            layer_2 = self.time_slot_MaxPool(layer_2).squeeze(2)
            collect = torch.cat((collect, layer_2.unsqueeze(0)), dim=0)
            
        last_two = collect[collect.size(0)-2:]
        
        collect = collect.permute(1, 0, 2)
        
        last_two, _ = self.last_two_GRU_1(last_two)
        last_two = self.last_two_Linear_1(last_two.contiguous().view(-1, last_two.size(2)))
        last_two = last_two.view(-1, 2, 256)
        
        layer_3, _ = self.week_GRU_1(collect)
        layer_3 = self.week_linear_1(layer_3.contiguous().view(-1, layer_3.size(2)))
        layer_3 = F.selu(self.norm_3(layer_3.view(-1, collect.size(1), 256)))
        
        layer_4 = (torch.cat((layer_3, last_two), dim=1))
        layer_4 = self.MaxPool(layer_4).squeeze(2)

        x = self.norm_4(layer_4)
        x = self.drop(x)
        x = self.highway_1(x)
        x = self.drop(x)
        x = self.linear_5(x)
        
        return x