import torch
import torch.nn as nn

#class SimpleNet(nn.Module):
#    def __init__(self, name, fc=0):
#        super(SimpleNet, self).__init__()
#        self.name = name
#        self.num_fc = fc
#        
#        self.params = {}
#        def _create_block(n_in, n_out, stride, name):
#            self.params[name+'/conv1'] \
#                = nn.Conv2d(n_in, n_out, 3, stride=1, padding=1, bias=False)
#            self.params[name+'/bn1'] \
#                = nn.BatchNorm2d(n_out)
#
#        _create_block(3, 64, 2, name+'/c1')
#        _create_block(64, 64, 2, name+'/c2')
#        _create_block(64, 64, 2, name+'/c3')
#        _create_block(64, 64, 2, name+'/c4')
#
#        self.params['relu'] = nn.ReLU()
#        self.params['pool'] = nn.MaxPool2d(2)
#        if self.num_fc > 0:
#            self.params['fc'] = nn.Linear(64*5*5, self.num_fc)
#
#        for key, val in self.params.items():
#            self.add_module(key, val)
#
#    def forward(self, x):
#        def _apply_block(x, name):
#            x = self.params[name+'/conv1'](x)
#            x = self.params[name+'/bn1'](x)
#            x = self.params['relu'](x)
#            x = self.params['pool'](x)
#            return x
#        
#        x = _apply_block(x, self.name+'/c1')
#        x = _apply_block(x, self.name+'/c2')
#        x = _apply_block(x, self.name+'/c3')
#        x = _apply_block(x, self.name+'/c4')
#        x = x.view(x.size(0), -1)
#        
#        if self.num_fc > 0:
#            x = self.params['fc'](x)
#
#        return x

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self, fc=0):
        super(SimpleNet, self).__init__()
        self.num_fc = fc

        layers = [Block(3, 64)]
        for i in range(3):
            layers.append(Block(64, 64))
        
        self.backbone = nn.Sequential(*layers)
        self.fc = nn.Linear(64*5*5, self.num_fc)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
