import torch
import torch.nn as nn
from lib.simplenet import SimpleNet



import pdb

class ProtoNet(object):
    def __init__(self,m name, nway, kshot, qsize, isTr,
            mbsize=1, arch='simple', config=None):
        self.name = name
        self.nway = nway
        self.kshot = kshot 
        self.qsize = qsize
        self.mbsize = mbsize

        self.arch = arch
        self.config = config

        if arch=='simple':
            self.base_cnn = SimpleNet(name)

    def forward(self, inputs):
        # first assume that we don't have metabatch:
        # sx.shape: 
        sx, sy, qx, qy = inputs
