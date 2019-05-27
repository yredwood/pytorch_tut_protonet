import torch
import torch.nn as nn
import time
import pdb


vgg_specs = {\
    'vgg11': ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
    'vgg13': ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
    'vgg16': ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
    'vgg19': ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])
    }

class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class VGGNet(nn.Module):
    def __init__(self, net_type='vgg16', fc=0, imgsize=320):
        super(VGGNet, self).__init__()
        self.num_fc = fc
        self.net_type = net_type

        vggnet_layers = []
        n_in = 3
        layers, filters = vgg_specs[self.net_type]
        for _layer, _filter in zip(layers, filters):
            vggnet_layers.append(ConvBlock(n_in, _filter))
            n_in = _filter
            for _ in range(_layer-1):
                vggnet_layers.append(ConvBlock(_filter, _filter))
            vggnet_layers.append(nn.MaxPool2d(2))
        
        self.backbone = nn.Sequential(*vggnet_layers)
        
        hidden_hw = imgsize
        for _ in range(len(layers)):
            hidden_hw = hidden_hw // 2
    
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_hw**2*filters[-1], 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, fc)
        self.relu = nn.ReLU()

    def forward(self, x):
        t0 = time.time()
        x = self.backbone(x)
        x = x.view(x.size(0), -1)


        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        print ('forward called in {:.3f} sec'.format(time.time()-t0))
        return x
