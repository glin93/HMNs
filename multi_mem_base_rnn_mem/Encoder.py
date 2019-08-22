"""
此处的Encoder直接copy原文的,
"""
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
import torch
from utils.config import *
from .Decoder import AttrProxy
import numpy as np
class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask,birnn=True):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
            if birnn ==True:
                rnn_cell = nn.GRU(embedding_dim, int(embedding_dim / 2) if birnn == True else embedding_dim,
                                  bidirectional=birnn, dropout=dropout)

                self.add_module("R_{}".format(hop), rnn_cell)
        self.birnn = birnn
        print('encoder rnn:{}'.format(self.birnn))
        if birnn == True:
            self.R = AttrProxy(self,"R_")
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
    def get_state(self,bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return torch.zeros(bsz, self.embedding_dim).cuda()
        else:
            return torch.zeros(bsz, self.embedding_dim)
    def forward(self,story):
        story = story.transpose(0,1)
        story_size = story.size()
        if self.unk_mask and self.training:
            ones = np.ones(story_size)
            # TODO:Check rand_mask
            rand_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            ones[:, :, 0] = ones[:, :, 0] * rand_mask
            a=torch.Tensor(ones)
            if USE_CUDA: a = a.cuda()
            story = story * a.long()
        u = [self.get_state(story.size(0))]
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1).long()) # b * (m * s) * e
            #TODO:check input shape
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            m_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
            if self.birnn:
                m_A, _ = self.R[hop](m_A)
            # 每个词有三个属性,【词，说话者,第几轮说的】相当于将一个词又拓展成三个词了
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob = self.softmax(torch.sum(m_A * u_temp, 2))
            embed_C = self.C[hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            m_C = torch.sum(embed_C, 2).squeeze(2)
            if self.birnn:
                m_C, _ = self.R[hop + 1](m_C)
            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return u_k
