import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils.config import *
import logging
import datetime
import ast
from utils.until_temp import entityList
import warnings
import copy
import numpy as np

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


MEM_TOKEN_SIZE = 3


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {UNK_token: 'UNK', PAD_token: "PAD", EOS_token: "EOS", SOS_token: "SOS"}
        self.n_words = 4  # Count default tokens

    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, src_seq, trg_seq, index_seq, gate_seq, src_word2id, trg_word2id, max_len, conv_seq, ent, ID,
                 kb_seq, kb_target_index, ):
        """Reads source and target sequences from txt files."""
        self.src_seqs = src_seq
        self.trg_seqs = trg_seq
        self.index_seqs = index_seq
        self.gate_seq = gate_seq
        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.max_len = max_len
        self.conv_seq = conv_seq
        self.ent = ent
        self.ID = ID
        self.kb_seq = kb_seq  # kb输入
        self.kb_target_index = kb_target_index  # 目标序列在kb中的位置

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]  # history_dialoge inputs
        trg_seq = self.trg_seqs[index]  # response target
        index_s = self.index_seqs[index]  # response_index_in_his_memory
        gete_s = self.gate_seq[index]
        kb_s = self.kb_seq[index]  # kb word inputs
        src_seq = self.preprocess(src_seq, self.src_word2id, trg=False)
        trg_seq = self.preprocess(trg_seq, self.trg_word2id)
        index_s = self.preprocess_inde(index_s, src_seq)
        gete_s = self.preprocess_gate(gete_s)
        conv_seq = self.conv_seq[index]
        conv_seq = self.preprocess(conv_seq, self.src_word2id, trg=False)
        kb_s = self.preprocess(kb_s, self.src_word2id, trg=False)  #
        kb_target_index = self.preprocess_inde(self.kb_target_index[index], kb_s)
        ID = self.ID[index]

        return src_seq, trg_seq, index_s, gete_s, self.max_len, self.src_seqs[index], self.trg_seqs[index], conv_seq, \
               self.ent[index], ID, kb_s, kb_target_index, self.kb_seq[index]

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        try:
            story = torch.Tensor(story)
        except:
            print(sequence)
            print(story)
        return story

    def preprocess_inde(self, sequence, src_seq):
        """Converts words to ids."""
        sequence = sequence + [len(src_seq) - 1]
        sequence = torch.Tensor(sequence)
        return sequence

    def preprocess_gate(self, sequence):
        """Converts words to ids."""
        sequence = sequence + [2]  # 这个EOS来自词表 0表示来自his,1表示来自kb,2表示来自vocab
        sequence = torch.Tensor(sequence)
        return sequence


def collate_fn(data):
    def merge(sequences, max_len):
        lengths = [len(seq) for seq in sequences]
        if (max_len):
            padded_seqs = torch.ones(len(sequences), max(lengths), MEM_TOKEN_SIZE).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end, :] = seq[:end]
        else:
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # seperate source and target sequences
    src_seqs, trg_seqs, ind_seqs, gete_s, max_len, src_plain, trg_plain, conv_seq, ent, ID, kb_seq, kb_target_index, \
    kb_plain = zip(*data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs, max_len)
    trg_seqs, trg_lengths = merge(trg_seqs, None)
    ind_seqs, _ = merge(ind_seqs, None)
    gete_s, _ = merge(gete_s, None)
    conv_seqs, conv_lengths = merge(conv_seq, max_len)

    kb_seq, kb_lengths = merge(kb_seq, max_len)
    kb_ind_seqs, _ = merge(kb_target_index, None)

    src_seqs = src_seqs.transpose(0, 1)
    trg_seqs = trg_seqs.transpose(0, 1)
    ind_seqs = ind_seqs.transpose(0, 1)
    kb_seq = kb_seq.transpose(0, 1)
    kb_ind_seqs = kb_ind_seqs.transpose(0, 1)

    gete_s = Variable(gete_s).transpose(0, 1)
    conv_seqs = Variable(conv_seqs).transpose(0, 1)

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        ind_seqs = ind_seqs.cuda()
        gete_s = gete_s.cuda()
        conv_seqs = conv_seqs.cuda()

        kb_seq = kb_seq.cuda()
        kb_ind_seqs = kb_ind_seqs.cuda()

    return src_seqs, src_lengths, trg_seqs, trg_lengths, ind_seqs, gete_s, src_plain, trg_plain, conv_seqs, conv_lengths, ent, ID, \
           kb_seq, kb_lengths, kb_ind_seqs, kb_plain


def read_langs(file_name, entity, max_line=None):
    logging.info(("Reading lines from {}".format(file_name)))
    data = []
    contex_arr = []
    conversation_arr = []
    conversation_sentence = []
    KB = []  # 仅包含KB信息
    u = None
    r = None
    user_counter = 0
    system_counter = 0
    system_res_counter = 0
    KB_counter = 0
    dialog_counter = 0
    turns_list = [] #统计每段对话的轮数
    history_list = [] #统计构建数据集时历史输入的长度
    warnings.warn('Using multi-mem')
    with open(file_name) as fin:
        cnt_ptr = 0
        cnt_total = 0
        max_r_len = 0
        cnt_lin = 1
        time_counter = 1
        for row_id, line in enumerate(fin):
            line = line.strip()
            if line:
                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r = line.split('\t')
                    if u != '<SILENCE>': user_counter += 1
                    system_counter += 1
                    # 这里针对每个词,给其加上是谁说的,第几轮说的
                    gen_u = generate_memory(u, "$u", str(time_counter))

                    conversation_sentence.append(gen_u)
                    # FIXME:Using collection.dequeue to fix this
                    if LOAD_LIMITS != None:
                        while len(conversation_sentence) > LOAD_LIMITS:
                            conversation_sentence.pop(0)
                    # Flatten the sentence
                    contex_arr = []
                    conversation_arr = []
                    for sentence in conversation_sentence:
                        for item in sentence:
                            conversation_arr.append(item)
                            contex_arr.append(item)

                    r_index = []
                    gate = []
                    for key in r.split(' '):
                        if ENTPTR:
                            warnings.warn('ENTPRT usage..')
                            if (key in entity):
                                index = [loc for loc, val in enumerate(contex_arr) if (val[0] == key)]
                                if (index):
                                    index = max(index)
                                    gate.append(0)
                                    cnt_ptr += 1
                                else:
                                    index = len(contex_arr)

                            else:
                                index = len(contex_arr)
                                gate.append(2)

                        else:
                            index = [loc for loc, val in enumerate(contex_arr) if (val[0] == key)]
                            if (index):
                                index = max(index)
                                gate.append(0)
                                cnt_ptr += 1
                            else:
                                index = len(contex_arr)
                                gate.append(2)
                        r_index.append(index)
                        system_res_counter += 1

                    if len(r_index) > max_r_len:
                        max_r_len = len(r_index)
                    contex_arr_temp = contex_arr + [['$$$$'] * MEM_TOKEN_SIZE]

                    ent = []
                    for key in r.split(' '):
                        if (key in entity):
                            ent.append(key)
                            # print(key)

                    #  将结束标记添加到KB Mem中去
                    if len(KB) == 0:
                        KB += [['$$$$'] * MEM_TOKEN_SIZE]
                    elif KB[-1] != ['$$$$'] * MEM_TOKEN_SIZE:
                        KB += [['$$$$'] * MEM_TOKEN_SIZE]

                    # 开始查找目标是否在KB中出现过,作为KB生成序列的目标
                    kb_target_index = []
                    for i, key in enumerate(r.split(' ')):
                        if ENTPTR:
                            # 在kb中迭代,查找key.
                            if key in entity:
                                index = [loc for loc, value in enumerate(KB) if value[0] == key]
                                if len(index) != 0:
                                    index = max(index)
                                    cnt_ptr += 1
                                    # 这个GATE已经在上面用过了.所以在这里只需要赋值即可
                                    gate[i] = 1
                                else:
                                    index = len(KB) - 1
                            else:
                                index = len(KB) - 1
                        else:
                            index = [loc for loc, val in enumerate(KB) if val[0] == key]
                            if len(index) != 0:
                                index = max(index)
                                gate[i] = 1
                                cnt_ptr += 1
                            else:
                                index = len(KB) - 1
                        kb_target_index.append(index)
                        cnt_total += 1

                    history_list.append(len(contex_arr_temp))
                    # 保存对话历史,contex_arr_temp是对话中的历史,历史结束位置会加上[['$$$$']*MEM_TOKEN_SIZE],只包括user的
                    # r 是针对这个历史的回复
                    # r_index 貌似是在历史中找到相同的词的位置,如果没出现过就指向历史中的最后的位置
                    # 说明的是回复中每个词是否在历史中出现过
                    # list(conversation_arr) 是历史对话的词的列表的列表,包括user和system
                    # ent 判断回复中是否出现了实体
                    # dialog_counter
                    data_dict = {
                        'history_inputs': contex_arr_temp,
                        'response': r,
                        'response_index_in_memory': r_index,
                        'response_index_in_memory_gate': gate,
                        'ent': ent,
                        'history': list(conversation_arr),
                        'kb_words': copy.deepcopy(KB),
                        'kb_target_index_in_memory': kb_target_index,
                        'id': copy.deepcopy(dialog_counter)
                    }
                    data.append(data_dict)
                    # data.append([contex_arr_temp, r, r_index, gate, list(conversation_arr), ent, dialog_counter])
                    gen_r = generate_memory(r, "$s", str(time_counter))

                    conversation_sentence.append(gen_r)

                    time_counter += 1
                else:
                    KB_counter += 1
                    r = line
                    if USEKB:
                        # contex_arr += generate_memory(r, "", "")
                        warnings.warn('Using kb information,but with out speaker information')
                        # 在添加新的KB前先判断下结尾是否有结束标记
                        #  将结束标记添加到KB Mem中去
                        if len(KB) == 0:
                            pass
                        elif KB[-1] == ['$$$$'] * MEM_TOKEN_SIZE:
                            KB.pop(-1)
                        KB += generate_memory(r, "", "")  # 仅包含KB信息
            else:
                cnt_lin += 1
                if (max_line and cnt_lin >= max_line):
                    break
                conversation_sentence = []
                if time_counter != 1:
                    turns_list.append(time_counter)
                time_counter = 1
                dialog_counter += 1
                KB = []
    max_len = max([len(d['history_inputs']) for d in data])
    logging.info("Pointer percentace= {} ".format(cnt_ptr / cnt_total))
    logging.info("Max responce Len: {}".format(max_r_len))
    logging.info("Max Input Len: {}".format(max_len))
    logging.info('Avg. History word inputs: {}'.format(np.mean(history_list)))
    logging.info("Avg. User Utterances: {}".format(user_counter * 1.0 / dialog_counter))
    logging.info("Avg. Bot Utterances: {}".format(system_counter * 1.0 / dialog_counter))
    logging.info("Avg. KB results: {}".format(KB_counter * 1.0 / dialog_counter))
    logging.info("Avg. responce Len: {}".format(system_res_counter * 1.0 / system_counter))
    logging.info('Avg. Dialoge turns: {}'.format(np.mean(turns_list)))
    print('Sample: ', data[1])
    return data, max_len, max_r_len


def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        for word in sent_token:
            temp = [word, speaker, 't' + str(time)] + ["PAD"] * (MEM_TOKEN_SIZE - 3)
            sent_new.append(temp)
    else:
        if sent_token[1] == "R_rating":
            sent_token = sent_token + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        else:
            sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new


def get_seq(pairs, lang, batch_size, type, max_len):
    x_seq = []
    y_seq = []
    ptr_seq = []
    gate_seq = []
    conv_seq = []
    ent = []
    ID = []  # what is this ?
    kb_seq = []
    kb_target_index = []
    for pair in pairs:
        x_seq.append(pair['history_inputs'])
        y_seq.append(pair['response'])
        ptr_seq.append(pair['response_index_in_memory'])
        gate_seq.append(pair['response_index_in_memory_gate'])
        ent.append(pair['ent'])
        conv_seq.append(pair['history'])
        kb_seq.append(pair['kb_words'])
        kb_target_index.append(pair['kb_target_index_in_memory'])
        ID.append(pair['id'])
        if (type):
            lang.index_words(pair['history_inputs'])
            lang.index_words(pair['response'], trg=True)
            lang.index_words(pair['kb_words'])

    dataset = Dataset(src_seq=x_seq, trg_seq=y_seq, index_seq=ptr_seq, gate_seq=gate_seq,
                      src_word2id=lang.word2index, trg_word2id=lang.word2index, max_len=max_len,
                      conv_seq=conv_seq, ent=ent, ID=ID, kb_seq=kb_seq, kb_target_index=kb_target_index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=collate_fn)
    return data_loader


def prepare_data_seq(task, batch_size=100, shuffle=True):
    if int(task) == 6:  # or  int(task) == 5:
         warnings.warn('Using kb before dialogues')
         file_train = 'data/dialog-bAbI-tasks/dialog-babi-task{}trnRP.txt'.format(task)
         file_dev = 'data/dialog-bAbI-tasks/dialog-babi-task{}devRP.txt'.format(task)
         file_test = 'data/dialog-bAbI-tasks/dialog-babi-task{}tstRP.txt'.format(task)
    else:
         file_train = 'data/dialog-bAbI-tasks/dialog-babi-task{}trn.txt'.format(task)
         file_dev = 'data/dialog-bAbI-tasks/dialog-babi-task{}dev.txt'.format(task)
         file_test = 'data/dialog-bAbI-tasks/dialog-babi-task{}tst.txt'.format(task)

    if (int(task) != 6):
        file_test_OOV = 'data/dialog-bAbI-tasks/dialog-babi-task{}tst-OOV.txt'.format(task)

    if int(task) != 6:
        ent = entityList('data/dialog-bAbI-tasks/dialog-babi-kb-all.txt', int(task))
    else:
        ent = entityList('data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-kb.txt', int(task))

    pair_train, max_len_train, max_r_train = read_langs(file_train, ent, max_line=None)
    pair_dev, max_len_dev, max_r_dev = read_langs(file_dev, ent, max_line=None)
    pair_test, max_len_test, max_r_test = read_langs(file_test, ent, max_line=None)

    max_r_test_OOV = 0
    max_len_test_OOV = 0
    if (int(task) != 6):
        pair_test_OOV, max_len_test_OOV, max_r_test_OOV = read_langs(file_test_OOV, ent, max_line=None)

    max_len = max(max_len_train, max_len_dev, max_len_test, max_len_test_OOV) + 1
    max_r = max(max_r_train, max_r_dev, max_r_test, max_r_test_OOV) + 1
    lang = Lang()

    # 把上面获得的文本变成序列
    train = get_seq(pair_train, lang, batch_size, True, max_len)
    dev = get_seq(pair_dev, lang, batch_size, False, max_len)
    test = get_seq(pair_test, lang, batch_size, False, max_len)
    if (int(task) != 6):
        testOOV = get_seq(pair_test_OOV, lang, batch_size, False, max_len)
    else:
        testOOV = []

    logging.info("Read %s sentence pairs train" % len(pair_train))
    logging.info("Read %s sentence pairs dev" % len(pair_dev))
    logging.info("Read %s sentence pairs test" % len(pair_test))
    if (int(task) != 6):
        logging.info("Read %s sentence pairs test" % len(pair_test_OOV))
    logging.info("Max len Input %s " % max_len)
    logging.info("Vocab_size %s " % lang.n_words)
    logging.info("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, testOOV, lang, max_len, max_r
