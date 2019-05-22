import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time

from lib.episode_generator import BatchGenerator
from lib.simplenet import SimpleNet


# ========= HYPERPARAMS ===========
data_dir = 'data/miniImagenet/train'
split = [('train', 'test'), (False, False)][0]
max_epoch = 100
batch_size = 128
num_workers = 4
# ========= HYPERPARAMS ===========


dtype = torch.float
device = torch.device('cuda:0')



train_dataset = BatchGenerator(data_dir, transform=True, split=split[0])
test_dataset = BatchGenerator(data_dir, split=split[1])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

train_net = SimpleNet('simplenet', fc=train_dataset.n_classes)
train_net.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(train_net.parameters(), lr=1e-2, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50,75], gamma=0.1)

# restoring
#train_net.load_only_possibles(torch.load('models/{}'.format(cfg)))
#train_net.load_state_dict(torch.load('models/{}'.format(cfg)), strict=False)


for epoch in range(max_epoch):
    accs, losses = [], []
    t0 = time.time()
    t2 = time.time()
    btime, mtime = 0, 0

    train_net.train()
    for i, batch in enumerate(train_dataloader):

        t1 = time.time()
        btime += t1 - t2

        x, y = batch
        x, y = x.to(device, dtype=dtype), y.to(device).long()

        pred = train_net.forward(x)
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
        x, y = x.to(device, dtype=dtype), y.to(device).long()
        train_net.eval()
        pred = train_net.forward(x)

        tacc = (pred.max(1)[1] == y).float().mean()
        taccs.append(tacc)

    taccs = torch.stack(taccs)
    accs = torch.stack(accs)
    losses = torch.stack(losses)

    print ('===Epoch {} / {} ====='.format(epoch, max_epoch))
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

    
    if (epoch % 10 == 0 and epoch != 0) or epoch == max_epoch-1:
        cfg = 'miniImagenet'
        loc = 'models/{}'.format(cfg)
        torch.save(train_net.state_dict(), loc)
        print ('saved in {}'.format(loc))




#
