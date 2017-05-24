import torch as th
from treeLSTM import TreeLSTM
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from load_data import QADataset
from torch.utils.data import DataLoader
import params
from progressbar import Percentage, Bar, ProgressBar, Timer
import datetime
import math


best_validation = 0.0
best_epoch = -1
batchsize = 100
trainset = QADataset('/home/xuehongyang/workspace/videoqa17/coco/'
                     'dataset/two_val_sets/train_binary_tree.pkl',
                     '/home/xuehongyang/workspace/videoqa17/coco/'
                     'dataset/two_val_sets/train_answer_3000.pkl')
valset = QADataset('/home/xuehongyang/workspace/videoqa17/coco/'
                   'dataset/two_val_sets/val_binary_tree.pkl',
                   '/home/xuehongyang/workspace/videoqa17/coco/'
                   'dataset/two_val_sets/val_answer_3000.pkl')


train_sample = trainset.__len__()
val_sample = valset.__len__()
train_loader = DataLoader(trainset, num_workers=2, batch_size=batchsize,
                          shuffle=True, pin_memory=True)
val_loader = DataLoader(valset, num_workers=2, batch_size=batchsize,
                        shuffle=False, pin_memory=True)
treelstm = TreeLSTM(2048, 2048, 2048, 3001)
val_batches = math.ceil(val_sample // batchsize)
train_batches = math.ceil(train_sample // batchsize)

if params.use_cuda:
    treelstm.cuda()
n_epoch = 30
optimizer = th.optim.Adam(treelstm.parameters(), lr=0.0001)
nn.utils.clip_grad_norm(treelstm.parameters(), 1.0)

output_interval = 20000

print('Start training with batch size %d' % batchsize)
print('TreeLSTM')
print('Dynamic batching vqa2.0 3000 answers')
print('Enabled:')
if params.two_fc:
    print(' two fc layers, normalized CNN')
else:
    print(' one fc layer')
if params.use_mult:
    print(' use mult in attention')
else:
    pass
if params.word_level_attention:
    print(' word level attention')
print(' add val1 into training')
for epoch in range(n_epoch):
    treelstm.train()
    widgets = ['Epoch: %d ' % epoch, Percentage(), ' ', Bar(marker='#'),
               ' ', Timer(), ' ']
    pbar = ProgressBar(widgets=widgets,
                       maxval=trainset.__len__()//batchsize).start()
    train_loss = 0
    for i, (v, q, op, a) in enumerate(train_loader):
        optimizer.zero_grad()
        current_bs = q.size()[0]
    
        v = Variable(v)
        q = Variable(q)
        a = Variable(a)
        if params.use_cuda:
            v = v.cuda()
            a = a.cuda()
            q = q.cuda()
        r = treelstm(v, q, op, current_bs)

        loss = F.cross_entropy(r, a.squeeze()).squeeze()

        train_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        pbar.update(i)
        # if i % output_interval == 0:
        #     print(' batch %d: loss = %f' %
        #           (i,
        #            train_loss / (i + 1)))

    t_loss = train_loss / (train_sample / batchsize)
    pbar.finish()
    print('Training loss = %f' % (t_loss))
    treelstm.eval()
    d1 = datetime.datetime.now()
    accuracy = 0

    for i, (v, q, op, a) in enumerate(val_loader):
        bs = q.size()[0]
        v = Variable(v)
        a = Variable(a)
        q = Variable(q)
        if params.use_cuda:
            v = v.cuda()
            a = a.cuda()
            q = q.cuda()
        r = treelstm(v, q, op, bs)
        r = r.max(dim=1)[1].squeeze()
        ac = th.eq(r, a.squeeze()).float().sum()
        ac = th.clamp(ac / 3.0, max=1.0)
        ac = ac.sum()
        accuracy += ac.data[0]
    d2 = datetime.datetime.now()
    duration = (d2 - d1).seconds
    print('validation takes %d seconds' % duration)
    print('validation accuracy = %f' % (accuracy / val_sample))
    th.save(treelstm.state_dict(),
            './snapshot_%d' % (epoch))

    if (best_validation < accuracy / val_sample):
        best_validation = accuracy / val_sample
        best_epoch = epoch
        th.save(treelstm.state_dict(),
                './snapshot_treelstm')
    if epoch > 1:
        print('best validation accuracy = %d, %f' % (
            best_epoch, best_validation))

