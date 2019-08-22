"""
这里包括两个Decoder类,DecoderrMemNN与DecoderrKBMemNN
为了保持不作过多的更改,KBMem在 MemNN中使用
"""
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from utils.config import *
import torch
import numpy as np


class DecoderrMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask, kb_mem_hop, debug=False, birnn=True):
        super(DecoderrMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        self.debug = debug
        # 修改方式1: 输入共享词向量,不同层用不同的RNN
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
            if birnn == True:
                rnn_cell = nn.GRU(embedding_dim, int(embedding_dim / 2) if birnn == True else embedding_dim,
                                  bidirectional=birnn, dropout=dropout)
                self.add_module("R_{}".format(hop), rnn_cell)

        self.C = AttrProxy(self, "C_")
        if birnn == True:
            self.R = AttrProxy(self, "R_")
        self.birnn = birnn
        print('usring birnn :{}'.format(self.birnn))
        self.softmax = nn.Softmax(dim=1)
        self.W1 = nn.Linear(2 * embedding_dim, self.num_vocab)
        self.gru = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.kb_memory = DecoderrKBMemNN(vocab, embedding_dim, hop=kb_mem_hop, dropout=dropout, unk_mask=unk_mask,
                                         debug=debug)

    def load_memory(self, story):
        # 提前保留矩阵A
        story_size = story.size()
        if self.unk_mask and self.training:
            ones = np.ones((story_size[0], story_size[1], story_size[2]))
            rand_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            ones[:, :, 0] = ones[:, :, 0] * rand_mask
            a = torch.Tensor(ones)
            if USE_CUDA:
                a = a.cuda()
            story = story * a.long()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1))  # .long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
            if self.birnn == True:
                embed_A, _ = self.R[hop](embed_A)
            m_A = embed_A
            embed_C = self.C[hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            if self.birnn == True:
                embed_C, _ = self.R[hop + 1](embed_C)
            m_C = embed_C
            self.m_story.append(m_A)
        self.m_story.append(m_C)

    def ptrMemDecoder(self, enc_query, last_hidden):
        """
        这里的RNN是按单布执行的,需要两个输入,RNN的输入enc_query,以及上一时刻的hidden_state
        :param enc_query:
        :param last_hidden:
        :return:p_kb,   [ batch x kb_len ]
                p_memory,[ batch x  history_size ]
                p_vocab,
                hidden
        """
        embed_q = self.C[0](enc_query)  # b * e
        output, hidden = self.gru(embed_q.unsqueeze(0), last_hidden)
        temp = []
        u = [hidden[0].squeeze()]  # [ batch_size, hidden_size]
        if self.debug:
            debug_dict = {}
            debug_att_list = []
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if (len(list(u[-1].size())) == 1): u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_lg = torch.sum(m_A * u_temp, 2)
            prob_ = self.softmax(prob_lg)  # batch x  history_len
            if self.debug:
                # only see batch 0
                debug_att_list.append(prob_[0].clone().reshape(1, -1))
            m_C = self.m_story[hop + 1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            ### modify this ,the output of history should concat with the kb history
            if (hop == 0):
                p_vocab = self.W1(torch.cat((u[0], o_k), 1))
            u_k = u[-1] + o_k
            u.append(u_k)
        if self.debug:
            temp_att = torch.cat(debug_att_list, dim=0).detach().cpu().numpy()  # hop_number  x his_len
            debug_dict['his_decoder_attn'] = temp_att
        p_memory = prob_lg
        if self.debug:
            p_kb, kb_final_output_vector, kb_debug_dict = self.kb_memory.ptrMemQuery(u[1])
            debug_dict.update(kb_debug_dict)
        else:
            p_kb, kb_final_output_vector = self.kb_memory.ptrMemQuery(u[-1])


        kb_switch_probality = 1  # torch.sigmoid( self.select_mem_linear_for_kb(concat_vector))
        memory_switch_probality = 1  # torch.sigmoid(self.select_mem_linear_for_memory(concat_vector))
        # state 应该包括,当前memory的输出,kb的输出,以及RNN的输出
        if self.debug:
            return p_kb, p_memory, p_vocab, (memory_switch_probality, kb_switch_probality), hidden, (
            u[-1], kb_final_output_vector, u[0]), debug_dict
        else:
            return p_kb, p_memory, p_vocab, (memory_switch_probality, kb_switch_probality), hidden, (
            u[-1], kb_final_output_vector, u[0])


class DecoderrKBMemNN(nn.Module):
    """
    单独存储KB的Memmory
    """

    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask, debug=False):
        super(DecoderrKBMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask

        for hop in range(self.max_hops + 1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)

            self.add_module("C_{}".format(hop), C)

        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.debug = debug

    def load_memory(self, story):
        # 提前保留矩阵A
        story_size = story.size()
        if self.unk_mask and self.training:
            ones = np.ones((story_size[0], story_size[1], story_size[2]))
            rand_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            ones[:, :, 0] = ones[:, :, 0] * rand_mask
            a = torch.Tensor(ones)
            if USE_CUDA:
                a = a.cuda()
            story = story * a.long()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1))  # .long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e

            m_A = embed_A
            embed_C = self.C[hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)

            m_C = embed_C
            self.m_story.append(m_A)
        self.m_story.append(m_C)

    def ptrMemQuery(self, enc_query):
        """
        :param enc_query: batch_size x hidden_size
        :return:
        """
        # TODO:Check here
        temp = []
        if self.debug:
            debug_dict = {}
            debug_att_list = []
        u = [enc_query.squeeze()]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if (len(list(u[-1].size())) == 1): u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_lg = torch.sum(m_A * u_temp, 2)
            prob_ = self.softmax(prob_lg)
            if self.debug:
                debug_att_list.append(prob_[0].clone().reshape(1, -1))
            m_C = self.m_story[hop + 1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        if self.debug:
            temp_att = torch.cat(debug_att_list, dim=0).detach().cpu().numpy()  # hop_number  x kb_len
            debug_dict['kb_decoder_attn'] = temp_att
        p_ptr = prob_lg
        if self.debug:
            return p_ptr, u[1], debug_dict
        else:
            return p_ptr, u[1]


# TODO:Check this
class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
