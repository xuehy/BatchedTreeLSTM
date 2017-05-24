import torch as th
from treeMN import TreeMN
from torch.autograd import Variable
import random
from load_data_yesno import QADataset
import params
from progressbar import Percentage, Bar, ProgressBar, Timer
import six.moves.cPickle as Pickle
import json

label2answer = None
with open('/home/xuehongyang/workspace/videoqa17/'
          'coco/dataset/label2answer_vqa2.pkl',
          'rb') as l2a:
    label2answer = Pickle.load(l2a)

testset = QADataset('/home/xuehongyang/workspace/videoqa17/coco/'
                    'dataset/test.pkl',
                    'test2015', 'test',
                    '/home/xuehongyang/workspace/videoqa17/coco/'
                    'dataset/test_type.pkl')


test_sample = testset.__len__()
treemn = TreeMN(512, 1024, 1024, 1001)
if params.use_cuda:
    treemn.cuda()
treemn.load_state_dict(th.load('snapshot'))
print('Start testing...')
treemn.eval()
widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#'),
           ' ', Timer(), ' ']
pbar = ProgressBar(widgets=widgets, maxval=testset.__len__()).start()

result = []
for i in range(test_sample):
    v, q, t, qid = testset.__getitem__(i)
    v = Variable(v, volatile=True)
    if params.use_cuda:
        v = v.cuda()
        
    r = treemn.forward(v, q)
    if t == 0:
        r = r[:, 0:1000]
        r = r.max(dim=1)[1].squeeze()
    else:
        r = r[:, -2:]
        r = r.max(dim=1)[1].squeeze()
        r += 998

    ans_l = r.data[0]
    ans = None
    if ans_l == 1000:
        ans = label2answer[random.randint(0, 999)]
    else:
        ans = label2answer[r.data[0]]
    result += [{'answer': ans, 'question_id': qid}]
    pbar.update(i)
pbar.finish()

result = json.dumps(result)
with open('result.json', 'w') as f:
    f.write(result)
