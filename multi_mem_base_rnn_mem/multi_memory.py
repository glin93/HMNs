import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from utils.masked_cross_entropy import *
from utils.config import *
import random
import numpy as np
from utils.measures import wer, moses_multi_bleu
from utils.config import PAD_token
import time
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from .Decoder import DecoderrMemNN
from .Encoder import EncoderMemNN
import numpy as np
import codecs

#from visdom import Visdom
Visdom=None
import torch
import warnings

torch.manual_seed(123)


class write_to_disk():
    def __init__(self, path):
        self.fp = codecs.open(path, 'w', 'utf8')
        print('result writing to ', path)

    def write(self, sentence):
        self.fp.write(sentence)

    def close(self):
        self.fp.close()


class Mem2Seq(nn.Module):
    # TODO:这里的 dropout 好像是不起作用的
    def __init__(self, hidden_size, max_len, max_r, lang, path, task, lr, n_layers, dropout, unk_mask, kb_mem_hop,
                 env_name='multimem',
                 debug=False, debug_host_ip='http://10.15.62.15',
                 enbirnn=False, debirnn=False):
        super(Mem2Seq, self).__init__()
        assert kb_mem_hop != None
        self.name = 'MultiMem2SeqRNN'
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.max_r = max_r
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.unk_mask = unk_mask
        self.debug = debug
        if path:
            if USE_CUDA:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th')
                self.decoder = torch.load(str(path) + '/dec.th')
            else:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
                self.decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)
            self.decoder.debug = self.debug
            self.decoder.kb_memory.debug = self.debug
        else:
            self.encoder = EncoderMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask, birnn=enbirnn)
            self.decoder = DecoderrMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask,
                                         debug=self.debug, kb_mem_hop=kb_mem_hop,
                                         birnn=debirnn
                                         )
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)
        self.loss = 0
        self.loss_memory = 0
        self.loss_vocabulary = 0
        self.loss_kb = 0
        self.print_every = 1
        self.batch_size = 0
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()
        try:
            self.vis = Visdom(server=debug_host_ip, env=env_name)
            print('server ip:{},env_name:{}'.format(debug_host_ip, env_name))
        except:
            print("Can't not use visdom ")
            self.vis = None


    def print_loss(self):
        self.print_loss_avg = self.loss / self.print_every
        self.print_loss_memory = self.loss_memory / self.print_every
        self.print_loss_kb = self.loss_kb / self.print_every
        self.print_loss_vocabulary = self.loss_vocabulary / self.print_every
        self.print_every += 1
        return 'Loss:{:.2f}, KB:{:.2f}, Memory:{:.2f}, Vocab:{:.2f}'.format(self.print_loss_avg, self.print_loss_kb,
                                                                            self.print_loss_memory,
                                                                            self.print_loss_vocabulary)

    def save_model(self, dec_type):
        name_data = "KVR/" if self.task == '' else "BABI/"
        directory = 'save/mem2seq-' + name_data + str(self.task) + 'HDD' + str(self.hidden_size) + 'BSZ' + str(
            args['batch']) + 'DR' + str(self.dropout) + 'L' + str(self.n_layers) + 'lr' + str(self.lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')

        return directory

    def _masked(self, target_tensors, inputs):
        """
        根据输入的pad标志对目标tensor进行填充为负无限
        :param target_tensors:  batch x lens
        :param inputs:         lens  x batch
        :return:
        """
        pad_matrix = torch.eq(inputs.transpose(0, 1), PAD_token)
        masked_tensor = target_tensors.masked_fill(pad_matrix, float('-50000'))
        # softmaxed_tensor = F.softmax(target_tensors,dim=1)
        return masked_tensor

    def _infer_get_next_in(self, memory_pro, kb_pro, vocab_pro, inputs, kb_inputs, input_lengths, kb_lengths):
        """
        :param memory_pro: batch x his_len
        :param kb_pro:  batch x kb_lens
        :param vocab_pro:  batch x vocab_szie
        :param inputs:表格
        :param kb_inputs:
        :param input_lengths:
        :param kb_lengths:
        :return:
        """
        batch_size = memory_pro.shape[0]
        _, topvi = vocab_pro.data.topk(1)
        kb_top_probality, kb_top_probality_pos = kb_pro.data.topk(1)
        kb_top_probality_inputs = torch.gather(kb_inputs, 0, kb_top_probality_pos.view(1, -1))
        memory_top_probality, memory_top_probality_pos = memory_pro.data.topk(1)
        memory_top_probality_inputs = torch.gather(inputs, 0, memory_top_probality_pos.view(1, -1))
        infer_pos_switch = []  # 0-> history, 1-> kb,2->vocab
        # for visualizing
        kb_cancate_history = torch.cat([memory_pro, kb_pro], dim=1)  # batch x lens
        next_in = []
        for i in range(batch_size):
            memory_pos_value = memory_top_probality_pos.squeeze()[i].item()
            kb_pos_value = kb_top_probality_pos.squeeze()[i].item()

            # warnings.warn('Using memory first strategy')
            # if memory_pos_value < input_lengths[i] - 1:
            #     next_in.append(memory_top_probality_inputs.squeeze()[i].item())
            #     infer_pos_switch.append((0, memory_pos_value))
            # elif kb_pos_value < kb_lengths[i] - 1:
            #     next_in.append(kb_top_probality_inputs.squeeze()[i].item())
            #     infer_pos_switch.append((1, kb_pos_value))
            # else:
            #     next_in.append(topvi.squeeze()[i].item())
            #     infer_pos_switch.append((2, next_in[-1]))

            warnings.warn('Using memory compare strategy')
            # 1) KB为end momery不为end: 用memory
            if kb_pos_value == kb_lengths[i] - 1 and memory_pos_value < input_lengths[i] - 1:
                next_in.append(memory_top_probality_inputs.squeeze()[i].item())
                infer_pos_switch.append((0, memory_pos_value))
            # 2) KB不为end memory 不为end:用概率最高的
            elif kb_pos_value < kb_lengths[i] - 1 and memory_pos_value < input_lengths[i] - 1:
                if memory_top_probality.squeeze()[i].item() < kb_top_probality.squeeze()[i].item():
                    next_in.append(kb_top_probality_inputs.squeeze()[i].item())
                    infer_pos_switch.append((1, kb_pos_value))
                else:
                    next_in.append(memory_top_probality_inputs.squeeze()[i].item())
                    infer_pos_switch.append((0, memory_pos_value))
            # 3) kb不为end , mem为end
            elif kb_pos_value < kb_lengths[i] - 1 and memory_pos_value == input_lengths[i] - 1:
                next_in.append(kb_top_probality_inputs.squeeze()[i].item())
                infer_pos_switch.append((1, kb_pos_value))
            else:
                next_in.append(topvi.squeeze()[i].item())
                infer_pos_switch.append((2, next_in[-1]))
        return next_in, infer_pos_switch, kb_cancate_history

    def train_batch(self, input_batches, input_lengths, target_batches, target_lengths, target_index, target_gate,
                    batch_size, clip,
                    teacher_forcing_ratio, conv_seqs, conv_lengths, kb_seqs, kb_lengths, kb_target_index, kb_plain,
                    reset):
        """
        TODO:Check shape of inputs
        :param input_batches: seq_len x batch_size x MEM_SIZE ,i.e: torch.Size([37, 2, 5])
        :param input_lengths:
        :param target_batches:
        :param target_lengths:
        :param target_index:
        :param target_gate:
        :param batch_size:
        :param clip:
        :param teacher_forcing_ratio:
        :param conv_seqs:
        :param conv_lengths:
        :param kb_seqs:  lens x batch x mem_size torch.Size([41, 2, 5])
        :param kb_lengths:
        :param reset:
        :return:
        """
        if reset:
            self.loss = 0
            self.loss_memory = 0
            self.loss_vocabulary = 0
            self.loss_kb = 0
            self.print_every = 1
        self.batch_size = batch_size
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0, 1))
        self.decoder.kb_memory.load_memory(kb_seqs.transpose(0, 1))
        decoder_input = torch.LongTensor([SOS_token] * batch_size)
        max_target_length = max(target_lengths)
        # store the output of
        all_decoder_outputs_vocab = torch.zeros(max_target_length, batch_size, self.output_size)
        all_decoder_outputs_memory = torch.zeros(max_target_length, batch_size, input_batches.size(0))
        all_decoder_outputs_kb = torch.zeros(max_target_length, batch_size, kb_seqs.size(0))  # 这里给出的是对位置的概率,而不是词表的概率！！！
        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_memory = all_decoder_outputs_memory.cuda()
            all_decoder_outputs_kb = all_decoder_outputs_kb.cuda()
            decoder_input = decoder_input.cuda()
        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        if use_teacher_forcing:
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_pkb, decoder_pmemory, decoder_vacab, switch_probality, decoder_hidden, pg_state = self.decoder.ptrMemDecoder(
                    decoder_input, decoder_hidden)
                # 先mask fill ,然后再使用softmax
                decoder_pmemory_normalized = self._masked(decoder_pmemory, input_batches[:, :, 0])
                decoder_pkb_normalized = self._masked(decoder_pkb, kb_seqs[:, :, 0])
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_memory[t] = decoder_pmemory_normalized
                all_decoder_outputs_kb[t] = decoder_pkb_normalized
                decoder_input = target_batches[t]  # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()
        else:
            for t in range(max_target_length):
                decoder_pkb, decoder_pmemory, decoder_vacab, switch_probality, decoder_hidden, pg_state = self.decoder.ptrMemDecoder(
                    decoder_input, decoder_hidden)
                decoder_pmemory_normalized = self._masked(decoder_pmemory, input_batches[:, :, 0])
                decoder_pkb_normalized = self._masked(decoder_pkb, kb_seqs[:, :, 0])
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_memory[t] = decoder_pmemory_normalized
                all_decoder_outputs_kb[t] = decoder_pkb_normalized
                next_in, _, _ = self._infer_get_next_in(memory_pro=decoder_pmemory_normalized,
                                                        kb_pro=decoder_pkb_normalized,
                                                        vocab_pro=decoder_vacab,
                                                        inputs=input_batches[:, :, 0],
                                                        kb_inputs=kb_seqs[:, :, 0],
                                                        input_lengths=input_lengths,
                                                        kb_lengths=kb_lengths)
                decoder_input = torch.LongTensor(next_in)  # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()
        # Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss_memory = masked_cross_entropy(
            all_decoder_outputs_memory.transpose(0, 1).contiguous(),  # -> batch x seq
            target_index.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss_kb = masked_cross_entropy(
            all_decoder_outputs_kb.transpose(0, 1).contiguous(),  # -> batch x seq
            kb_target_index.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss = loss_Vocab + loss_memory + loss_kb
        loss.backward()
        # Clip gradient norms
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_memory += loss_memory.item()
        self.loss_vocabulary += loss_Vocab.item()
        self.loss_kb += loss_kb.item()

    def evaluate_batch(self, batch_size, input_batches, input_lengths, target_batches, target_lengths, target_index,
                       target_gate,
                       src_plain,
                       conv_seqs,
                       conv_lengths,
                       kb_seqs,
                       kb_lengths,
                       kb_target_index,
                       kb_plain,
                       step,
                       Analyse=False):
        """
        FIXME 可视化每个Mem 位置上面的score 分数
        :param batch_size:
        :param input_batches:
        :param input_lengths:
        :param target_batches:
        :param target_lengths:
        :param target_index:
        :param target_gate:
        :param src_plain:
        :param conv_seqs:
        :param conv_lengths:
        :param kb_seqs:   kb_len x batch x Mem_size
        :param kb_lengths:
        :param kb_target_index:  res_len x batch
        :param kb_plain:kb 纯文本
        :param step 当前训练第几步
        :param epoch 当前训练第几轮
        :return:
        """
        # if np.equal(np.array(kb_lengths),1).sum() >0:
        #     print('kb lengths more than 1')
        # print(kb_lengths)
        self.encoder.train(False)
        self.decoder.train(False)
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0, 1))
        self.decoder.kb_memory.load_memory(kb_seqs.transpose(0, 1))
        decoder_input = torch.LongTensor([SOS_token] * batch_size)
        decoded_words = []
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
        p = []
        for elm in src_plain:
            elm_temp = [word_triple[0] for word_triple in elm]
            p.append(elm_temp)
        kb_p = []
        for elm in kb_plain:
            elm_temp = [word_triple[0] for word_triple in elm]
            kb_p.append(elm_temp)
        self.from_whichs = []
        atten_history = []  # 保存每个time-step的attention score的结果
        if self.debug:
            max_r_dict_list = []
        for t in range(self.max_r):
            # decoder_pkb batch_size x kb_len,
            # decoder_pmemory batch_size x his_len ,
            # decoder_vacab  batch_size x total_vocab
            if self.debug:
                decoder_pkb, decoder_pmemory, decoder_vacab, switch_probality, decoder_hidden, pg_state, debug_dict = self.decoder.ptrMemDecoder(
                    decoder_input, decoder_hidden)
            else:
                decoder_pkb, decoder_pmemory, decoder_vacab, switch_probality, decoder_hidden, pg_state = self.decoder.ptrMemDecoder(
                    decoder_input, decoder_hidden)
            decoder_pmemory_normalized = self._masked(decoder_pmemory, input_batches[:, :, 0])
            decoder_pkb_normalized = self._masked(decoder_pkb, kb_seqs[:, :, 0])
            # This is for switch probabiltiy, but WE don't use it anymore in current version .
            # if self.debug:
            #    print(switch_probality[0].squeeze()[0].item(),switch_probality[0].squeeze()[1].item())
            next_in, infer_pos_switch, kb_cancate_history = self._infer_get_next_in(
                memory_pro=decoder_pmemory_normalized,
                kb_pro=decoder_pkb_normalized,
                vocab_pro=decoder_vacab,
                inputs=input_batches[:, :, 0],
                kb_inputs=kb_seqs[:, :, 0],
                input_lengths=input_lengths,
                kb_lengths=kb_lengths)
            decoder_input = torch.LongTensor(next_in)  # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()
            temp = []
            from_which = []
            for row_index, (source_index, pos_id) in enumerate(infer_pos_switch):
                if source_index == 0:
                    temp.append(p[row_index][pos_id])
                    from_which.append('p')
                elif source_index == 1:
                    temp.append(kb_p[row_index][pos_id])
                    from_which.append('p')
                elif source_index == 2:
                    if pos_id == EOS_token:
                        temp.append('<EOS>')
                    else:
                        temp.append(self.lang.index2word[pos_id])
                    from_which.append('v')
                else:
                    raise RuntimeError()
            decoded_words.append(temp)
            self.from_whichs.append(from_which)
            # 注意这里是history 在前,kb在后
            # sending to visdom for visulazing
            if self.debug:
                max_r_dict_list.append(debug_dict)
                atten_history.append(kb_cancate_history[0, :].tolist())

        self.from_whichs = np.array(self.from_whichs)
        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        return decoded_words  # , acc_ptr, acc_vac

    def evaluate(self, dev, avg_best, epoch, BLEU=False, Analyse=False, type='dev'):
        # Analyse 是在分析Mem中每个attn的score使用的
        assert type == 'dev' or type == 'test'
        logging.info("STARTING EVALUATION:{}".format(type))
        acc_avg = 0.0
        wer_avg = 0.0
        bleu_avg = 0.0
        acc_P = 0.0
        acc_V = 0.0
        microF1_PRED, microF1_PRED_cal, microF1_PRED_nav, microF1_PRED_wet = [], [], [], []
        microF1_TRUE, microF1_TRUE_cal, microF1_TRUE_nav, microF1_TRUE_wet = [], [], [], []
        ref = []
        hyp = []
        ref_s = ""
        hyp_s = ""
        dialog_acc_dict = {}
        pbar = tqdm(enumerate(dev), total=len(dev))
        if Analyse == True:
            write_fp = write_to_disk('./multi-mem-generate.txt')
            # 统计有多少数据是从memory中复制出来的

        for j, data_dev in pbar:
            words = self.evaluate_batch(batch_size=len(data_dev[1]),
                                        input_batches=data_dev[0],
                                        input_lengths=data_dev[1],
                                        target_batches=data_dev[2],
                                        target_lengths=data_dev[3],
                                        target_index=data_dev[4],
                                        target_gate=data_dev[5],
                                        src_plain=data_dev[6],
                                        conv_seqs=data_dev[-6],
                                        conv_lengths=data_dev[-5],
                                        kb_seqs=data_dev[-4],
                                        kb_lengths=data_dev[-3],
                                        kb_target_index=data_dev[-2],
                                        kb_plain=data_dev[-1],
                                        step=j,
                                        Analyse=Analyse)
            # acc_P += acc_ptr
            # acc_V += acc_vac
            acc = 0
            w = 0
            temp_gen = []
            for i, row in enumerate(np.transpose(words)):
                st = ''
                for e in row:
                    if e == '<EOS>':
                        break
                    else:
                        st += e + ' '
                temp_gen.append(st)
                correct = data_dev[7][i]
                ### compute F1 SCORE
                if args['dataset'] == 'kvr':
                    # TODO:Check this
                    f1_true, f1_pred = computeF1(data_dev[8][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE += f1_true
                    microF1_PRED += f1_pred
                    f1_true, f1_pred = computeF1(data_dev[9][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE_cal += f1_true
                    microF1_PRED_cal += f1_pred
                    f1_true, f1_pred = computeF1(data_dev[10][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE_nav += f1_true
                    microF1_PRED_nav += f1_pred
                    f1_true, f1_pred = computeF1(data_dev[11][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE_wet += f1_true
                    microF1_PRED_wet += f1_pred
                elif args['dataset'] == 'babi' and int(self.task) == 6:
                    f1_true, f1_pred = computeF1(data_dev[-6][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE += f1_true
                    microF1_PRED += f1_pred
                if args['dataset'] == 'babi':
                    if data_dev[-5][i] not in dialog_acc_dict.keys():
                        dialog_acc_dict[data_dev[-5][i]] = []
                    if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                        acc += 1
                        dialog_acc_dict[data_dev[-5][i]].append(1)
                    else:
                        dialog_acc_dict[data_dev[-5][i]].append(0)
                else:
                    if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                        acc += 1
                # print("Correct:"+str(correct.lstrip().rstrip()))
                # print("\tPredict:"+str(st.lstrip().rstrip()))
                # print("\tFrom:"+str(self.from_whichs[:,i]))
                # w += wer(correct.lstrip().rstrip(), st.lstrip().rstrip())
                ref.append(str(correct.lstrip().rstrip()))
                hyp.append(str(st.lstrip().rstrip()))
                ref_s += str(correct.lstrip().rstrip()) + "\n"
                hyp_s += str(st.lstrip().rstrip()) + "\n"

            # write batch data to disk
            if Analyse == True:
                for gen, gold in zip(temp_gen, data_dev[7]):
                    write_fp.write(gen + '\t' + gold + '\n')

            acc_avg += acc / float(len(data_dev[1]))
            wer_avg += w / float(len(data_dev[1]))
            pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg / float(len(dev)),
                                                            wer_avg / float(len(dev))))

        if Analyse == True:
            write_fp.close()

        # dialog accuracy
        if args['dataset'] == 'babi':
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            logging.info("Dialog Accuracy:\t" + str(dia_acc * 1.0 / len(dialog_acc_dict.keys())))
            self._send_metrics(epoch, type, acc=dia_acc * 1.0 / len(dialog_acc_dict.keys()))
        if args['dataset'] == 'kvr':
            f1 = f1_score(microF1_TRUE, microF1_PRED, average='micro')
            f1_cal = f1_score(microF1_TRUE_cal, microF1_PRED_cal, average='micro')
            f1_wet = f1_score(microF1_TRUE_wet, microF1_PRED_wet, average='micro')
            f1_nav = f1_score(microF1_TRUE_nav, microF1_PRED_nav, average='micro')
            logging.info("F1 SCORE:\t" + str(f1))
            logging.info("F1 CAL:\t" + str(f1_cal))
            logging.info("F1 WET:\t" + str(f1_wet))
            logging.info("F1 NAV:\t" + str(f1_nav))
            self._send_metrics(epoch, type, f1=f1, f1_cal=f1_cal, f1_wet=f1_wet, f1_nav=f1_nav)
        elif args['dataset'] == 'babi' and int(self.task) == 6:
            f1 = f1_score(microF1_TRUE, microF1_PRED, average='micro')
            logging.info("F1 SCORE:\t" + str(f1))
            self._send_metrics(epoch, type, babi_6_f1=f1)
        # Report Bleu score
        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        # Report Acc per response
        self._send_metrics(epoch, type, acc_response=acc_avg / float(len(dev)))
        logging.info("BLEU SCORE:" + str(bleu_score))

        if Analyse == False:
            # Send loss
            self._send_metrics(epoch, type, total_loss=self.print_loss_avg,
                               ptr_loss=self.print_loss_kb,
                               vocab_loss=self.print_loss_vocabulary,
                               his_loss=self.print_loss_memory,
                               bleu_score=bleu_score)

            if (BLEU):
                if (bleu_score >= avg_best and bleu_score != 0):
                    if type == 'dev':
                        directory = self.save_model(str(self.name) + str(bleu_score))
                        locals_var = locals()

                        logging.info("MODEL SAVED")
                return bleu_score
            else:
                acc_avg = acc_avg / float(len(dev))
                if (acc_avg >= avg_best):
                    if type == 'dev':
                        locals_var = locals()
                        directory = self.save_model(str(self.name) + str(acc_avg))
                        logging.info("MODEL SAVED")
                return acc_avg
        else:
            if (BLEU):
                return bleu_score
            else:
                return acc_avg

    def _send_metrics(self, epoch, type, **krgs):
        def show(win, epoch, value):
            opts = {}
            opts['title'] = win
            if self.vis.win_exists(win):
                self.vis.line(X=np.array([epoch]), Y=np.array([value]), win=win, update='append', opts=opts)
            else:
                self.vis.line(X=np.array([epoch]), Y=np.array([value]), win=win, opts=opts)

        if self.vis:
            for key, value in krgs.items():
                show(type + key, epoch, value)


def computeF1(entity, st, correct):
    y_pred = [0 for z in range(len(entity))]
    y_true = [1 for z in range(len(entity))]
    # 如果在生成的句子中,
    for k in st.lstrip().rstrip().split(' '):
        if (k in entity):
            y_pred[entity.index(k)] = 1
    return y_true, y_pred
