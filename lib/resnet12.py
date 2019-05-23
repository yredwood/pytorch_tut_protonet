import torch
import torch.nn as nn
#
#class ResNet12(nn.Module):
#    def __init__(self, name, fc=0):
#        super(ResNet12, self).__init__()
#        self.name = name
#        self.num_fc = fc
#        
#        self.params = {}
#
#        def _create_block(n_in, n_out, name):
#            self.params[name+'/c0'] \
#                = nn.Conv2d(n_in, n_out, 3, stride=1, padding=1, bias=False)
#
#            self.params[name+'/b0'] \
#                = nn.BatchNorm2d(n_out)
#
#            for i in range(1,3):
#                self.params[name+'/c{}'.format(i)] \
#                    = nn.Conv2d(n_out, n_out, 3, stride=1, 
#                            padding=1, bias=False)
#
#                self.params[name+'/b{}'.format(i)] \
#                    = nn.BatchNorm2d(n_out)
#
#            self.params[name+'/shortcut'] \
#                = nn.Conv2d(n_in, n_out, 1, stride=1, padding=0, bias=False)
#
#        _create_block(3, 64, 'G0')
#        _create_block(64, 128, 'G1')
#        _create_block(128, 256, 'G2')
#        _create_block(256, 512, 'G3')
#
#        self.params['relu'] = nn.LeakyReLU(0.1)
#        self.params['pool'] = nn.MaxPool2d(2, padding=1)
#        self.params['dropout'] = nn.Dropout(0.1)
#        self.params['avgpool'] = nn.AdaptiveAvgPool2d((1,1))
#
#        if self.num_fc > 0:
#            self.params['fc'] = nn.Linear(512, self.num_fc)
#
#        for key, val in self.params.items():
#            self.add_module(key, val)
#
#    def forward(self, x):
#        def _apply_block(x, name):
#            shortcut = self.params[name+'/shortcut'](x)
#            for i in range(3):
#                x = self.params[name+'/c{}'.format(i)](x)
#                x = self.params[name+'/b{}'.format(i)](x)
#                x = self.params['relu'](x)
#            x = x + shortcut
#            x = self.params['pool'](x)
#            x = self.params['dropout'](x)
#            return x
#
#        x = _apply_block(x, 'G0')
#        x = _apply_block(x, 'G1')
#        x = _apply_block(x, 'G2')
#        x = _apply_block(x, 'G3')
#        x = self.params['avgpool'](x)
#        x = x.view(x.size(0), -1)
#
#        if self.num_fc > 0:
#            x = self.params['fc'](x)
#
#        return x

class ResBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_out)

        self.conv2 = nn.Conv2d(n_out, n_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_out)

        self.conv3 = nn.Conv2d(n_out, n_out, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(n_out)
    
        self.shortcut_conv = nn.Conv2d(n_in, n_out, 1, stride=1, padding=0, bias=False)

        self.relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d(2, padding=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = x + shortcut
        x = self.pool(x)
        x = self.dropout(x)
        return x
        


class ResNet12(nn.Module):
    def __init__(self, fc=0):
        super(ResNet12, self).__init__()
            
        self.num_fc = fc 
        layers = []
        layers.append(ResBlock(3, 64))
        layers.append(ResBlock(64, 128))
        layers.append(ResBlock(128, 256))
        layers.append(ResBlock(256, 512))
        self.backbone = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if self.num_fc > 0:
            self.fc = nn.Linear(512, self.num_fc)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.num_fc > 0:
            x = self.fc(x)
        return x
