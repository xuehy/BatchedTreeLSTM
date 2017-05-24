import torch as th
import torch.nn as nn
import torch.nn.functional as F
import params
from torch.nn import init


class TreeLSTMCell(nn.Module):
    def __init__(self, feature_dim):
        super(TreeLSTMCell, self).__init__()
        # parameter for tree lstm cell
        self.Ui = nn.Linear(feature_dim, feature_dim)
        self.Uf = nn.Linear(feature_dim, feature_dim)
        self.Uo = nn.Linear(feature_dim, feature_dim)
        self.Uu = nn.Linear(feature_dim, feature_dim)

        nn.init.uniform(self.Ui.weight, -0.01, 0.01)
        nn.init.uniform(self.Uf.weight, -0.01, 0.01)
        nn.init.uniform(self.Uo.weight, -0.01, 0.01)
        nn.init.uniform(self.Uu.weight, -0.01, 0.01)
        self.Ui.bias.data.fill_(0)
        self.Uf.bias.data.fill_(1)
        self.Uo.bias.data.fill_(0)
        self.Uu.bias.data.fill_(0)

    # non-leaf node do not have input
    def forward(self, c1, c2, h1, h2):
        h = h1 + h2
        i = F.sigmoid(self.Ui(h))
        o = F.sigmoid(self.Uo(h))
        u = F.tanh(self.Uu(h))
        f1 = F.sigmoid(self.Uf(h1))
        f2 = F.sigmoid(self.Uf(h2))
        c = i * u + f1 * c1 + f2 * c2
        h = o * F.tanh(c)
        return h, c


class TreeLSTM(nn.Module):
    def __init__(self, visual_dim, word_dim,
                 feature_dim, num_output):
        super(TreeLSTM, self).__init__()
        self.treeunint = TreeLSTMCell(feature_dim)
        self.feature_dim = feature_dim
        self.visual_dim = visual_dim
        self.wembed = nn.Embedding(params.word_number,
                                   params.word_dim)

        self.Wo = nn.Linear(word_dim, feature_dim)
        self.Wu = nn.Linear(word_dim, feature_dim)
        self.Wi = nn.Linear(word_dim, feature_dim)

        self.WQ = nn.Linear(feature_dim, feature_dim)
        self.WV = nn.Linear(feature_dim, feature_dim, bias=False)
        self.WP = nn.Linear(feature_dim, 1)

        self.FC1 = nn.Linear(feature_dim, feature_dim)
        self.FC2 = nn.Linear(feature_dim, num_output)
        self.word_dim = word_dim
        self.W = nn.Linear(visual_dim, feature_dim)

        nn.init.uniform(self.wembed.weight, -1, 1)
        self.Wo.bias.data.fill_(0)
        self.Wu.bias.data.fill_(0)
        self.Wi.bias.data.fill_(0)
        self.WQ.bias.data.fill_(0)
        self.WP.bias.data.fill_(0)
        self.FC1.bias.data.fill_(0)
        self.FC2.bias.data.fill_(0)
        self.W.bias.data.fill_(0)
        nn.init.uniform(self.Wo.weight, -0.01, 0.01)
        nn.init.uniform(self.Wu.weight, -0.01, 0.01)
        nn.init.uniform(self.Wi.weight, -0.01, 0.01)
        nn.init.uniform(self.W.weight, -0.01, 0.01)
        nn.init.uniform(self.WQ.weight, -0.01, 0.01)
        nn.init.uniform(self.WV.weight, -0.01, 0.01)
        nn.init.uniform(self.WP.weight, -0.01, 0.01)
        nn.init.uniform(self.FC1.weight, -0.01, 0.01)
        nn.init.uniform(self.FC2.weight, -0.01, 0.01)

    # q: bs x feature_dim
    # image_embeddings: bs x num_region x feature_dim
    def attention(self, q, image_embeddings):
        hA = None
        # (bs * num_region) x feature_dim
        wvi = self.WV(image_embeddings.view(-1,
                                            self.feature_dim))
        wvi = wvi.view(-1, params.num_region, self.feature_dim)
        wqq = self.WQ(q).unsqueeze(1).expand_as(image_embeddings)
        if not params.use_mult:
            hA = F.tanh(wqq + wvi)
        else:
            hA = F.tanh(wqq * wvi)

        # hA: (bs * num_region) x feature_dim -> bs x n_re
        hA = hA.view(-1, self.feature_dim)
        hA = self.WP(hA).squeeze()
        hA = hA.view(-1, params.num_region)
        p = F.softmax(hA)
        # p: bs x n_re
        weighted = p.unsqueeze(2).expand_as(
            image_embeddings) * image_embeddings
        v = weighted.sum(dim=1)
        return v + q

    def dynamicBatching(self, buff_c, buff_h, ops, images, bs):
        # iterate over buffs
        stacks = [[] for i in range(bs)]
        for i in range(params.max_sequence_length):
            h_lefts, c_lefts = [], []
            h_rights, c_rights = [], []
            bc = buff_c[:, i]
            bh = buff_h[:, i]
            op = ops[:, i]
            for j in range(bs):
                stack = stacks[j]
                if op[j] == 0:  # shift
                    stack.append((bh[j], bc[j]))
                elif op[j] == 1:  # reduce
                    right = stack.pop()
                    h_rights.append(right[0])
                    c_rights.append(right[1])
                    left = stack.pop()
                    h_lefts.append(left[0])
                    c_lefts.append(left[1])
                else:
                    pass
            if c_lefts:
                h, c = self.treeunint(th.stack(tuple(c_lefts)),
                                      th.stack(tuple(c_rights)),
                                      th.stack(tuple(h_lefts)),
                                      th.stack(tuple(h_rights)))
                h, c = th.split(th.cat(h), self.feature_dim),\
                    th.split(th.cat(c), self.feature_dim)
                h, c = iter(h), iter(c)
                for j in range(bs):
                    stack = stacks[j]
                    if op[j] == 1:
                        stack.append((next(h), next(c)))
        return th.stack([stack.pop()[0] for stack in stacks])

    # sent: bs x max_sequence_length
    # v: bs x num_region x visual_dim
    def forward(self, v, questions, ops, bs):
        v = v.view(-1, self.visual_dim)
        v = F.tanh(self.W(v))
        v = v.view(-1, params.num_region, self.feature_dim)
        # sent: bs x max_sequence_length x word_dim
        questions = self.wembed(questions)
        # the input of the treeLSTM
        i = F.sigmoid(self.Wi(questions.view(-1, self.feature_dim)))
        u = F.tanh(self.Wu(questions.view(-1, self.feature_dim)))
        o = F.sigmoid(self.Wo(questions.view(-1, self.feature_dim)))
        buff_c = i * u
        buff_h = o * F.tanh(buff_c)
        buff_c, buff_h = buff_c.view(-1, params.max_sequence_length,
                                     self.feature_dim),\
            buff_h.view(-1, params.max_sequence_length, self.feature_dim)

        # buff_c, buff_h: initial input of the LSTM
        # r: bs x feature_dim
        r = self.dynamicBatching(buff_c, buff_h, ops, v, bs)
        r = self.attention(r, v).squeeze()
        if params.two_fc:
            return self.FC2(F.tanh(self.FC1(r)))
        else:
            return self.FC2(r)
