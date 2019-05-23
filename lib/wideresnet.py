import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, stride):
        super(ResBlock, self).__init__()
        
        self.downsample = n_in != n_out
        self.bn1 = nn.BatchNorm2d(n_in)
        self.conv1 = nn.Conv2d(n_in, n_out, 3, 
                stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(n_out)
        self.conv2 = nn.Conv2d(n_out, n_out, 3, 
                stride=1, padding=1, bias=False)

        if self.downsample:
            self.shortcut_conv = nn.Conv2d(n_in, n_out, 1, 
                    stride=stride, padding=0, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        x = self.bn1(x)
        x = self.relu(x)
        if self.downsample:
            shortcut = self.shortcut_conv(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = x + shortcut
        
        return x
        
class WideResNet(nn.Module):
    def __init__(self, depth=16, filters=16, k=10, fc=0):
        super(WideResNet, self).__init__()
            
        self.num_fc = fc 
        self.depth = depth
        self.k = k
        self.stride = 3 # dependent on input image
        fb = [filters, filters*k, filters*2*k, filters*4*k]
        assert (depth-4)%6 == 0
        self.n_blocks = [(depth-4)//6] * 3


        self.prebn1 = nn.BatchNorm2d(3)
        self.preconv = nn.Conv2d(3, fb[0], 3, 
                stride=1, padding=1, bias=False)
        self.prebn2 = nn.BatchNorm2d(fb[0])

        layers = []
        layers.append(ResBlock(fb[0], fb[1], 1))
        layers.append(ResBlock(fb[1], fb[1], 1))

        layers.append(ResBlock(fb[1], fb[2], self.stride))
        layers.append(ResBlock(fb[2], fb[2], 1))

        layers.append(ResBlock(fb[2], fb[3], self.stride))
        layers.append(ResBlock(fb[3], fb[3], 1))

        self.backbone = nn.Sequential(*layers)

        self.postbn = nn.BatchNorm2d(fb[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        if self.num_fc > 0:
            self.fc = nn.Linear(fb[3], self.num_fc)

    def forward(self, x):
        x = self.prebn1(x)
        x = self.preconv(x)
        x = self.prebn2(x)

        x = self.backbone(x)
        x = self.postbn(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.num_fc > 0:
            x = self.fc(x)
        return x
