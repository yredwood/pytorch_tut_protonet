import torch
import torch.cuda as cuda
import GPUtil

class QueueIterator:
    def __init__(self, q):
        self.q = q 

    def __iter__(self):
        return self

    def __next__(self):
        thing = self.q.get()

        if thing is None:
            raise StopIteration()

        return thing


def enqueue_loader_output(batch_queue, control_queue, loader):
    while True:
        ctrl = control_queue.get_nowait()
        print (ctrl)
        if ctrl is False:
            break

        for batch_id, batch in enumerate(loader):
            batch_queue.put((batch_id, batch))

        batch_queue.put(None)

def get_available_gpu_ids(order='first'):
    deviceIDs = GPUtil.getAvailable(order=order, 
            limit=8, 
            maxLoad = 0.5,
            maxMemory = 0.5,
            includeNan=False,
            excludeID=[], excludeUUID=[])
    return deviceIDs


def mixup(x, y, lam, yto1hot=0):
    # yto1hot -> n_classes
    y1hot = torch.zeros(y.size(0), yto1hot)
    y1hot[torch.arange(y.size(0)), y] = 1
    y = y1hot
    perm = torch.randperm(x.size(0))
    x_perm = x[perm]
    y_perm = y[perm]
    x_mx = lam*x + (1-lam)*x_perm
    y_mx = lam*y + (1-lam)*y_perm
    return x_mx, y_mx

def xent(pred, y):
    return -(pred.log_softmax(dim=-1) * y).sum(dim=-1).mean()
    

if __name__=='__main__':
    print (get_available_gpu_ids('first'))



#
