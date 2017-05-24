import torch as th
import six.moves.cPickle as Pickle
from torch.utils.data import Dataset
import numpy as np
import params


class QADataset(Dataset):
    def __init__(self, filepath, answerpath):
        # the trees
        self.data = None
        with open(filepath, 'rb') as datafile:
            self.data = Pickle.load(datafile)

        self.length = len(self.data)

        self.answer = None
        with open(answerpath, 'rb') as answerfile:
            self.answer = Pickle.load(answerfile)
        self.word_dict = None
        # the word dict is a mapping of sentence word to one hot id
        with open('/home/xuehongyang/workspace/videoqa17/coco/'
                  'dataset/word_dict.pkl', 'rb') as embed:
            self.word_dict = Pickle.load(embed)
        self.word_dict['unk'] = params.word_number - 1

    def __len__(self):
        return self.length

    def operation(self, word):
        if word == ')':
            return 1
        else:
            return 0

    def label(self, word):
        if word in self.word_dict.keys():
            return self.word_dict[word]
        else:
            return self.word_dict['unk']

    def __getitem__(self, index):
        sample = self.data[index]
        qid = sample[0]
        vid = sample[1]
        ques = sample[2]
        op = [self.operation(i) for i in ques]
        ques = [self.label(i) for i in ques]
        if len(op) < params.max_sequence_length:
            op += [2 for i in range(params.max_sequence_length - len(op))]
            ques += [params.word_number - 1 for i in range(
                params.max_sequence_length -
                len(ques))]
        ans = self.answer[qid]
        vf = Pickle.load(open(
            '/home/xuehongyang/workspace/videoqa17/coco/dataset/'
            'res5c/COCO_000000%06d.pkl' %
            (int(vid)),
            'rb'),
                         encoding='latin1')
        b = np.sqrt((vf * vf).sum(axis=1)) + 1e-15
        vf = vf / b.reshape(49, 1)
        return th.from_numpy(np.asarray(vf, dtype=np.float32)),\
            th.from_numpy(np.asarray(ques, dtype=np.int64)),\
            th.from_numpy(np.asarray(op, dtype=np.int64)),\
            th.from_numpy(np.asarray([ans], dtype=np.int64))
