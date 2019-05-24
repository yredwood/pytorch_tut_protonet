import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time

import pdb

from lib.episode_generator import MiniImagenet
from lib.episode_generator import CIFAR10
from lib.episode_generator import Chexpert
from lib.episode_generator import ImagenetTrain, ImagenetVal

#from lib.simplenet import SimpleNet
#from lib.resnet12 import ResNet12
#from lib.wideresnet import WideResNet
from lib.vggnet import VGGNet

from lib.utils import get_available_gpu_ids
from lib.utils import xent
from lib.utils import mixup
from lib.utils import xent


# ========= HYPERPARAMS ===========
def parse_args():
    parser = argparse.ArgumentParser(description='imgnet size')
    parser.add_argument('--project', dest='project', default='pytorchnets')
    parser.add_argument('--datatype', dest='datatype', default='imagenet')
    parser.add_argument('--max_epoch', dest='max_epoch', default=20, type=int)
    parser.add_argument('--lr_decay', dest='lr_decay', default=0.2, type=float)
    parser.add_argument('--lr_decay_list', dest='lr_decay_list', default='60,120,160')
    parser.add_argument('--lr_init', dest='lr_init', default=1e-2, type=float)
    parser.add_argument('--num_workers', dest='num_workers', default=32, type=int)
    parser.add_argument('--num_gpus', dest='num_gpus', default=1, type=int)
    parser.add_argument('--per_gpu_batch', dest='per_gpu_batch', default=16, type=int)
    parser.add_argument('--aug_mixup', dest='aug_mixup', default=1, type=int)
    parser.add_argument('--arch', dest='arch', default='vgg')
    parser.add_argument('--wd', dest='weight_decay', default=5e-4, type=float)
    parser.add_argument('--pr', dest='pretrain', default=0, type=int)

    args = parser.parse_args()
    return args
# ========= HYPERPARAMS ===========


if __name__=='__main__':
    args = parse_args()
    print ('='*30, 'ARGS', '='*30)
    args.batch_size = args.per_gpu_batch * args.num_gpus 
    args.devices = get_available_gpu_ids()[:args.num_gpus]
    args.lr_decay_list = [int(r) for r in args.lr_decay_list.split(',')]
    args.save_str = 'net.{}_data.{}_task.{}_mixup.{}_batchsize.{}'\
            .format(args.arch, args.datatype, 'clf', 
                    args.aug_mixup, args.batch_size)

    torch.cuda.set_device(args.devices[0])
    for arg in sorted(vars(args)):
        print ('%15s: %s'%(arg, getattr(args, arg)))
    print ('='*63)


    if args.datatype=='miniImagenet':
        data_dir = 'data/miniImagenet/train'
        train_dataset = MiniImagenet(data_dir, transform=True, split='train')
        test_dataset = MiniImagenet(data_dir, split='test')
    elif args.datatype=='cifar10':
        data_dir = 'data/cifar-10-batches-py'
        train_dataset = CIFAR10(data_dir, transform=True, phase='train')
        test_dataset = CIFAR10(data_dir, transform=False, phase='test')
    elif args.datatype=='chexpert':
        data_dir = 'data/CheXpert-v1.0-small'
        train_dataset = Chexpert(data_dir, transform=True, phase='train')
        test_dataset = Chexpert(data_dir, transform=False, phase='valid')
    elif args.datatype=='imagenet':
        data_dir = 'data/Imagenet'
        train_dataset = ImagenetTrain(data_dir)
        test_dataset = ImagenetVal(data_dir)



    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
            shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.num_workers)

    if args.arch=='simple':
        train_net = SimpleNet(fc=train_dataset.n_classes)
    elif args.arch=='res12':
        train_net = ResNet12(fc=train_dataset.n_classes)
    elif args.arch=='wdres':
        train_net = WideResNet(fc=train_dataset.n_classes)
    elif args.arch=='vgg':
        train_net = VGGNet(fc=train_dataset.n_classes, imgsize=train_dataset.hw)
    train_net = nn.DataParallel(train_net, args.devices).cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(train_net.parameters(), lr=args.lr_init, 
            momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            args.lr_decay_list, gamma=args.lr_decay)

    # restoring
    if args.pretrain:
        train_net.load_state_dict(torch.load('models/{}'.format(args.save_str)))
    #train_net.load_only_possibles(torch.load('models/{}'.format(cfg)))
    #tramport Queuein_net.load_state_dict(torch.load('models/{}'.format(cfg)), strict=False)

    for epoch in range(args.max_epoch):
        accs, losses = [], []
        t0 = time.time()
        t2 = time.time()
        btime, mtime = 0, 0

        train_net.train()
        for i, batch in enumerate(train_dataloader):

            t1 = time.time()
            btime += t1 - t2

            x, y = batch
            if args.aug_mixup:
                lam = np.random.beta(1, 1)
                if epoch >= args.max_epoch - 20:
                    lam = 1

                x, y = mixup(x, y, lam, train_dataset.n_classes)
                x, y = x.cuda(), y.long().cuda()
                pred = train_net.forward(x)
                loss = xent(pred, y)
                acc = (pred.max(1)[1] == y.max(1)[1]).float().mean()
            else:
                x, y = x.cuda(), y.cuda().long()
                pred = train_net.forward(x)
                if len(y.shpae) == 1:
                    y = y.max(1)[1]
                loss = loss_fn(pred, y)
                acc = (pred.max(1)[1] == y).float().mean()

            accs.append(acc)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t2 = time.time()
            mtime += t2 - t1
        
        lr_scheduler.step()
        taccs = []
        train_net.eval()
        for i, batch in enumerate(test_dataloader):
            x, y = batch 
            #x, y = x.to(device, dtype=dtype), y.to(device).long()
            x, y = x.cuda(non_blocking=True), y.long().cuda(non_blocking=True)
            train_net.eval()
            pred = train_net.forward(x)
            if len(y.shape) == 1:
                y = y.max(1)[1]
            tacc = (pred.max(1)[1] == y).float().mean()
            taccs.append(tacc)

        taccs = torch.stack(taccs)
        accs = torch.stack(accs)
        losses = torch.stack(losses)

        print ('===Epoch {} / {} ====='.format(epoch, args.max_epoch))
        print ('Trainin acc : {:.3f}  |  '
                'loss : {:.3f}  |  '
                'Test Acc: {:.3f} |'
                'Lr: {:.3f} * 1e-3 |'
                'in {:.3f} sec'\
                .format(torch.mean(accs)*100., 
                    torch.mean(losses), 
                    torch.mean(taccs)*100., 
                    optimizer.param_groups[0]['lr'] * 1e+3,
                    time.time()-t0))
        print ('profiling - batch time: {:.3f} / model time: {:.3f}  || proportion - {:.3f}'\
                .format(btime, mtime, btime/mtime*100.))

        
        if (epoch % 10 == 0 and epoch != 0) or epoch == args.max_epoch-1:
            loc = 'models/{}'.format(args.save_str)
            torch.save(train_net.state_dict(), loc)
            print ('saved in {}'.format(loc))




    #
