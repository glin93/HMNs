import torch.utils.data as data
import torch
from torch.autograd import Variable
import logging
import ast
from utils.config import *
import codecs
import numpy as np
MEM_TOKEN_SIZE =5
class Lang:
    # 建立语言的字典
    def __init__(self):
        self.word2index={}
        self.index2word = {UNK_token: 'UNK', PAD_token: "PAD", EOS_token: "EOS",  SOS_token: "SOS"}
        self.word2count={}
        self.n_words= 4
    def index_words(self,story,trg =False):
        # 有两种模式,第一种处理回复的str类型,第二种是处理历史的list
        if trg == True:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)
    def index_word(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] =1
            self.index2word[self.n_words]  = word
            self.n_words +=1
        else:
            self.word2count[word] +=1
class Dataset(data.Dataset):
    def __init__(self, src_seq, trg_seq, index_seq, gate_seq,src_word2id, trg_word2id,max_len,entity,entity_cal,entity_nav,
                 entity_wet, conv_seq,kb_seq,
                 kb_target_index):
        """Reads source and target sequences from txt files."""
        self.src_seqs = src_seq
        self.trg_seqs = trg_seq
        self.index_seqs = index_seq
        self.gate_seq = gate_seq
        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.max_len = max_len
        self.entity = entity
        self.entity_cal = entity_cal
        self.entity_nav = entity_nav
        self.entity_wet = entity_wet
        self.conv_seq = conv_seq
        self.kb_seq=  kb_seq
        self.kb_target_index = kb_target_index
    def __getitem__(self, index):
        """
        训练数据:
            两个Memory的输入:
                history_dialog
                kb_memory
            两个Memory的对应输出:
                his 目标词是否memory是否出现过
                kb  目标词是否memory是否出现过
            Decoder 输入
            Decoder 输出
        :param index:
        :return:
        """
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index] # history_dialoge inputs
        trg_seq = self.trg_seqs[index] # response target
        index_s = self.index_seqs[index]# response_index_in_his_memory
        gete_s = self.gate_seq[index]
        kb_s = self.kb_seq[index]
        src_seq = self.preprocess(src_seq, self.src_word2id, trg=False)
        trg_seq = self.preprocess(trg_seq, self.trg_word2id)
        index_s = self.preprocess_inde(index_s, src_seq) # response_index_in_his_memory
        gete_s = self.preprocess_gate(gete_s)
        conv_seq = self.conv_seq[index]  # 这个变量并没有什么用
        conv_seq = self.preprocess(conv_seq, self.src_word2id, trg=False) # 这个变量并没有什么用
        kb_s = self.preprocess(kb_s, self.src_word2id, trg=False)
        kb_target_index= self.preprocess_inde(self.kb_target_index[index] , kb_s)
        # 这个
        return src_seq, trg_seq, index_s, gete_s, self.max_len, self.src_seqs[index], self.trg_seqs[index], self.entity[
            index], self.entity_cal[index], self.entity_nav[index], self.entity_wet[index], conv_seq,kb_s,kb_target_index,self.kb_seq[index]
    def __len__(self):
        return self.num_total_seqs
    def preprocess(self,sequence,word2id,trg = True):
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i,word_triple in enumerate(sequence):
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
        """在序列问题上,trg序列在记忆上的位置需要多加一个对应 trg的EOS,即输入序列最后一个位置len(src_seq)-1"""
        sequence = sequence + [len(src_seq)-1]
        try:
            sequence = torch.Tensor(sequence)
        except:
            print(sequence)
        return sequence
    def preprocess_gate(self, sequence):
        """在序列问题上,trg序列在记忆上的位置需要多加一个对应 trg的EOS,即输入序列最后一个位置len(src_seq)-1"""
        sequence = sequence + [2] # 这个EOS来自词表 0表示来自his,1表示来自kb,2表示来自vocab
        sequence = torch.Tensor(sequence)
        return sequence
def collate_fn(data):
    def merge(sequences, max_len):
        lengths = [len(seq) for seq in sequences]
        if (max_len): # 这里的max_len 并不是提供一个具体的树枝
            padded_seqs = torch.ones(len(sequences), max(lengths), MEM_TOKEN_SIZE).long()
            #padded_seqs = torch.zeros(len(sequences), max(lengths), MEM_TOKEN_SIZE).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                try:
                    padded_seqs[i, :end, :] = seq[:end]
                except TypeError:
                    print('seq')
        else:
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
            #padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                try:
                    padded_seqs[i, :end] = seq[:end]
                except:
                    print('seq')
        return padded_seqs, lengths
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[-3]), reverse=True)
    # seperate source and target sequences
    src_seqs, trg_seqs, ind_seqs, gete_s, max_len, src_plain, trg_plain, entity, entity_cal, entity_nav, entity_wet, conv_seq,kb_seq,kb_target_index,kb_plain = zip(
        *data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs, max_len)
    trg_seqs, trg_lengths = merge(trg_seqs, None)
    ind_seqs, _ = merge(ind_seqs, None)
    gete_s, _ = merge(gete_s, None)
    conv_seqs, conv_lengths = merge(conv_seq, max_len)
    kb_seq,kb_lengths =  merge(kb_seq,max_len)
    kb_ind_seqs, _ = merge(kb_target_index,None)
    src_seqs = src_seqs.transpose(0, 1)
    trg_seqs = trg_seqs.transpose(0, 1)
    ind_seqs = ind_seqs.transpose(0, 1)
    gete_s = gete_s.transpose(0, 1)
    conv_seqs = conv_seqs.transpose(0, 1)
    kb_seq = kb_seq.transpose(0,1)
    kb_ind_seqs = kb_ind_seqs.transpose(0,1)
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        ind_seqs = ind_seqs.cuda()
        gete_s = gete_s.cuda()
        conv_seqs = conv_seqs.cuda()
        kb_seq = kb_seq.cuda()
        kb_ind_seqs =  kb_ind_seqs.cuda()

    return src_seqs, src_lengths, trg_seqs, trg_lengths, ind_seqs, gete_s, src_plain, trg_plain, entity, entity_cal, entity_nav, entity_wet, conv_seqs, conv_lengths,kb_seq,kb_lengths,kb_ind_seqs,kb_plain
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
def generate_memory(sent,speaker,time):
    sent_token = sent.split(' ')
    if speaker == '$u' or speaker == '$s':
        sent_new = [[word,speaker,'t'+str(time)] +['PAD']* (MEM_TOKEN_SIZE -3) for word in sent_token]
    else:
        # 这是KB的信息,本来的顺序是 subject,relation,object
        # 反一下之后就是object,relation,subject 这样就是实体在第一位了
        sent_new = [sent_token[::-1] + ['PAD']*(MEM_TOKEN_SIZE -len(sent_token))]
    return sent_new
def read_langs(file_name,max_line=None):
    # 从原数据中读取 对话,kb等等
    logging.info('Readling lines from {}'.format(file_name))
    data = []
    contex_arr = []  # 包含了对话历史与KB的信息
    conversation_arr = [] # 包含了对话历史的信息
    conversation_sentence = []
    KB = [] # 仅包含KB信息
    user_counter=0
    system_counter=0
    system_res_counter = 0
    max_r_len = 0
    cnt_ptr = 0
    cnt_voc = 0
    entity = {}
    KB_counter = 0
    cnt_lin =1
    dialog_counter = 0
    time_counter = 1
    turns_list = [] #统计每段对话的轮数
    history_list = [] #统计构建数据集时历史输入的长度
    with codecs.open(file_name,'r','utf8') as fp:
        for row_index, line in enumerate(fp):
            line = line .strip()
            if line :
                # 数据中以 # 表明对话人物类型
                if '#' in line:
                    line = line.replace('#','')
                    task_type=line
                    continue
                # 每行都有一个编号
                nid,line = line.split(' ',1)
                # 说明还在一个对话中
                if '\t' in line:
                    u,r,gold = line.split('\t')
                    user_counter+=1
                    system_counter+=1
                    # 将 u 中的词转换成对应的3个属性 ,词,谁说的,第几轮
                    gen_u  = generate_memory(u,'$u',str(nid))

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
                    # 开始查找目标是否在对话历史中出现过
                    for key in r.split(' '):
                        # 在对话历史中迭代,查找key.
                        index= [ loc  for loc,value in enumerate(contex_arr) if value[0] == key]
                        if len(index) !=0:
                            index = max(index)
                            gate.append(0)
                            cnt_ptr +=1
                        else:
                            index = len(contex_arr)
                            gate.append(2)
                        r_index.append(index)
                        system_res_counter +=1 # 记录每个回复的词数
                    # 比较得到回复的最大长度
                    if len(r_index) > max_r_len:
                        max_r_len = len(r_index)
                    # 生成训练数据的输入
                    contex_arr_temp = contex_arr + [['$$$$']*MEM_TOKEN_SIZE]
                    ent_index_calendar=[]
                    ent_index_navigation=[]
                    ent_index_weather =[]
                    # 每个回复后面是python列表的字符串,将其转换为python的对象
                    gold = ast.literal_eval(gold)
                    if task_type == 'weather':
                        ent_index_weather = gold
                    elif task_type == 'schedule':
                        ent_index_calendar = gold
                    elif task_type == 'navigate':
                        ent_index_navigation = gold
                    else:
                        raise RuntimeError('Not find task type')
                    ent_index = list(set(ent_index_calendar+ent_index_navigation+ent_index_weather))
                    # contex_arr不断记录对话历史
                    # 保存对话历史,contex_arr_temp是对话中的历史加上了 '$$$'结束标志,历史结束位置会加上[['$$$$']*MEM_TOKEN_SIZE]
                    # r 是针对这个历史的回复
                    # r_index 貌似是在历史中找到相同的词的位置,如果没出现过就指向历史中的最后的位置
                    # 说明的是回复中每个词是否在历史中出现过
                    # list(conversation_arr) 是历史对话的词的列表的列表,包括user和system
                    # ent 判断回复中是否出现了实体
                    # dialog_counter
                    # data_dict={
                    #     'history_with_kb':contex_arr_temp,
                    #     'response':r,
                    #     'response_index_in_memory':r_index,
                    #     'response_index_in_memory_gate':gate,
                    #     'ent_index':ent_index,
                    #     'ent_calendar':list(set(ent_index_calendar)),
                    #     'ent_navigation':list(set(ent_index_navigation)),
                    #     'ent_weather':list(set(ent_index_weather)),
                    #     'history': list(conversation_arr)
                    #
                    # }
                    # data .append(data_dict)
                    # 针对KB为空的情况,需要加上一个结束标记
                    if len(KB) == 0:
                        KB+=[['$$$$']*MEM_TOKEN_SIZE]
                    elif KB[-1] != ['$$$$']*MEM_TOKEN_SIZE:
                        KB += [['$$$$'] * MEM_TOKEN_SIZE]
                    # 开始查找目标是否在KB中出现过,作为KB生成序列的目标
                    kb_target_index = []
                    for i,key in enumerate(r.split(' ')):
                        # 在kb中迭代,查找key.
                        index= [ loc  for loc,value in enumerate(KB) if value[0] == key]
                        if len(index) !=0:
                            index = max(index)
                            cnt_ptr += 1
                            # 这个GATE已经在上面用过了.所以在这里只需要赋值即可
                            gate[i]=1
                        else:
                            index = len(KB) -1
                            cnt_voc += 1
                        kb_target_index.append(index)

                    history_list.append(len(contex_arr_temp))

                    data_dict={
                        'history_inputs':contex_arr_temp,
                        'response':r,
                        'response_index_in_memory':r_index,
                        'response_index_in_memory_gate':gate,
                        'ent_index':ent_index,
                        'ent_calendar':list(set(ent_index_calendar)),
                        'ent_navigation':list(set(ent_index_navigation)),
                        'ent_weather':list(set(ent_index_weather)),
                        'history': list(conversation_arr),
                        'kb_words':KB,
                        'kb_target_index_in_memory':kb_target_index
                    }
                    #data.append([contex_arr_temp,r,r_index,gate,ent_index,list(set(ent_index_calendar)),list(set(ent_index_navigation)),list(set(ent_index_weather)),list(conversation_arr),KB,kb_target_index])
                    data.append(data_dict)
                    gen_r = generate_memory(r,'$s',str(nid))

                    conversation_sentence.append(gen_r)

                    time_counter +=1 # for calculating stastic
                else:
                    # 没有\t说明是KB相关信息
                    KB_counter += 1
                    r= line
                    for e in line .split(' '):
                        entity[e]= 0
                    # TODO:这里的KB并没有加上结束标志
                    #contex_arr += generate_memory(r,'',str(nid))
                    KB += generate_memory(r,'',str(nid))
            else:
                # 说明新的对话要开始了,清空数据
                cnt_lin +=1
                entity = {}
                if max_line and cnt_lin > max_line:
                    break
                if time_counter !=1:
                    turns_list.append(time_counter)
                time_counter=0
                contex_arr =[]
                conversation_arr =[]
                conversation_sentence = []
                KB =[]
                dialog_counter +=1
        max_len = max([len(d['history_inputs']) for d in data])
        logging.info("Pointer percentace= {} ".format(cnt_ptr / (cnt_ptr + cnt_voc)))
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
def get_seq(pairs, lang, batch_size, type, max_len):
    """
                        data_dict={
                        'history_with_kb':contex_arr_temp,
                        'response':r,
                        'response_index_in_memory':r_index,
                        'response_index_in_memory_gate':gate,
                        'ent_index':ent_index,
                        'ent_calendar':list(set(ent_index_calendar)),
                        'ent_navigation':list(set(ent_index_navigation)),
                        'ent_weather':list(set(ent_index_weather)),
                        'history': list(conversation_arr),
                        'kb_words':KB,
                        'kb_target_index_in_memory':kb_target_index
                    }
    :param pairs:
    :param lang:
    :param batch_size:
    :param type:
    :param max_len:
    :return:
    """
    logging.info('type of get_seq :{}'.format(type))
    x_seq = []
    y_seq = []
    ptr_seq = []
    gate_seq = []
    entity = []
    entity_cal = []
    entity_nav = []
    entity_wet = []
    conv_seq = []
    kb_seq = []
    kb_target_index = []
    for pair in pairs:
        x_seq.append(pair['history_inputs'])
        y_seq.append(pair['response'])
        ptr_seq.append(pair['response_index_in_memory'])
        gate_seq.append(pair['response_index_in_memory_gate'])
        entity.append(pair['ent_index'])
        entity_cal.append(pair['ent_calendar'])
        entity_nav.append(pair['ent_navigation'])
        entity_wet.append(pair['ent_weather'])
        conv_seq.append(pair['history'])
        kb_seq.append(pair['kb_words'])
        kb_target_index.append(pair['kb_target_index_in_memory'])
        if (type):
            lang.index_words(pair['history_inputs'])
            lang.index_words(pair['response'], trg=True)
            lang.index_words(pair['kb_words'])
    dataset = Dataset(src_seq=x_seq,
                      trg_seq=y_seq,
                      index_seq=ptr_seq,
                      gate_seq=gate_seq,
                      src_word2id=lang.word2index,
                      trg_word2id=lang.word2index,
                      max_len =max_len,
                      entity=entity,
                      entity_cal=entity_cal,
                      entity_nav=entity_nav,
                      entity_wet=entity_wet,
                      conv_seq=conv_seq,
                      kb_seq=kb_seq,
                     kb_target_index=kb_target_index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=collate_fn)
    return data_loader
def prepare_data_seq(task='', batch_size=100):
    file_train = 'data/KVR/{}train.txt'.format(task)
    file_dev = 'data/KVR/{}dev.txt'.format(task)
    file_test = 'data/KVR/{}test.txt'.format(task)
    pair_train, max_len_train, max_r_train = read_langs(file_train, max_line=None)
    pair_dev, max_len_dev, max_r_dev = read_langs(file_dev, max_line=None)
    pair_test, max_len_test, max_r_test = read_langs(file_test, max_line=None)
    max_r_test_OOV = 0
    max_len_test_OOV = 0
    max_len = max(max_len_train, max_len_dev, max_len_test, max_len_test_OOV) + 1
    max_r = max(max_r_train, max_r_dev, max_r_test, max_r_test_OOV) + 1
    lang = Lang()
    train = get_seq(pair_train, lang, batch_size, True, max_len)
    dev = get_seq(pair_dev, lang, batch_size, False, max_len)
    test = get_seq(pair_test, lang, batch_size, False, max_len)
    logging.info("Read %s sentence pairs train" % len(pair_train))
    logging.info("Read %s sentence pairs dev" % len(pair_dev))
    logging.info("Read %s sentence pairs test" % len(pair_test))
    logging.info("Max len Input %s " % max_len)
    logging.info("Vocab_size %s " % lang.n_words)
    logging.info("USE_CUDA={}".format(USE_CUDA))
    # print(lang.index2word)
    return train, dev, test, [], lang, max_len, max_r
if __name__ == '__main__':
    read_langs('../data/KVR/test.txt',None)