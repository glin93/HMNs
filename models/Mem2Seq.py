import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
from utils.masked_cross_entropy import *
from utils.config import *
import random
import numpy as np
import datetime
from utils.measures import wer, moses_multi_bleu
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk
import os
from sklearn.metrics import f1_score
import codecs
# from visdom import Visdom
Visdom=None

class write_to_disk():

    def __init__(self, path):
        self.fp = codecs.open(path, 'w', 'utf8')
        print('result writing to ',path)
    def write(self, sentence):
        self.fp.write(sentence)

    def close(self):
        self.fp.close()


class Mem2Seq(nn.Module):
    def __init__(self, hidden_size, max_len, max_r, lang, path, task, lr, n_layers, dropout, unk_mask,env_name='singlemem',debug=False,debug_host_ip='http://10.15.62.15'):
        super(Mem2Seq, self).__init__()
        self.name = "Mem2Seq"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.max_len = max_len ## max input
        self.max_r = max_r ## max responce len        
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.unk_mask = unk_mask
        
        if path:
            if USE_CUDA:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.decoder = torch.load(str(path)+'/dec.th')
            else:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
        else:
            self.encoder = EncoderMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
            self.decoder = DecoderrMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer,mode='max',factor=0.5,patience=1,min_lr=0.0001, verbose=True)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1
        self.batch_size = 0
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()
        try:
            self.vis = Visdom(server=debug_host_ip,env=env_name)
        except:
            print("Can't not use visdom ")
            self.vis=None

    def print_loss(self):    
        self.print_loss_avg =  self.loss / self.print_every
        self.print_loss_ptr =  self.loss_ptr / self.print_every
        self.print_loss_vac =  self.loss_vac / self.print_every
        self.print_every += 1     
        return 'L:{:.2f}, VL:{:.2f}, PL:{:.2f}'.format(self.print_loss_avg,self.print_loss_vac,self.print_loss_ptr)
    
    def save_model(self, dec_type):
        name_data = "KVR/" if self.task=='' else "BABI/"
        directory = 'save/mem2seq-'+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+'L'+str(self.n_layers)+'lr'+str(self.lr)+str(dec_type)                 
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory+'/enc.th')
        torch.save(self.decoder, directory+'/dec.th')

        return directory
        
    def train_batch(self, input_batches, input_lengths, target_batches, 
                    target_lengths, target_index, target_gate, batch_size, clip,
                    teacher_forcing_ratio, conv_seqs, conv_lengths, reset):  

        if reset:
            self.loss = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.print_every = 1

        self.batch_size = batch_size
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss_Vocab,loss_Ptr= 0,0

        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0,1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        
        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = Variable(torch.zeros(max_target_length, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(max_target_length, batch_size, input_batches.size(0)))

        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            decoder_input = decoder_input.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        if use_teacher_forcing:    
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden  = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                decoder_input = target_batches[t]# Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()            
        else:
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                _, toppi = decoder_ptr.data.topk(1)
                _, topvi = decoder_vacab.data.topk(1)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                ## get the correspective word in input
                top_ptr_i = torch.gather(input_batches[:,:,0],0,Variable(toppi.view(1, -1)))
                next_in = [ self._get_value(top_ptr_i.squeeze()[i]) if(toppi.squeeze()[i] < input_lengths[i]-1) else topvi.squeeze()[i] for i in range(batch_size)]
                decoder_input = Variable(torch.LongTensor(next_in)) # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()
                  
        #Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy_for_origin_usage(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths
        )
        loss_Ptr = masked_cross_entropy_for_origin_usage(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(), # -> batch x seq
            target_index.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths
        )

        loss = loss_Vocab + loss_Ptr
        loss.backward()
        
        # Clip gradient norms
        if torch.__version__ >='0.4.0':
            ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
            dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        else:
            ec = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
            dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss +=  self._get_value(loss)
        self.loss_ptr += self._get_value(loss_Ptr)
        self.loss_vac += self._get_value(loss_Vocab)

    def evaluate_batch(self,batch_size,input_batches, input_lengths, target_batches, target_lengths, target_index,target_gate,src_plain, conv_seqs, conv_lengths):  
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)  
        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0,1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(self.max_r, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(self.max_r, batch_size, input_batches.size(0)))
        #all_decoder_outputs_gate = Variable(torch.zeros(self.max_r, batch_size))
        # Move new Variables to CUDA

        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            #all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
            decoder_input = decoder_input.cuda()
        
        p = []
        for elm in src_plain:
            elm_temp = [ word_triple[0] for word_triple in elm ]
            p.append(elm_temp) 
        
        self.from_whichs = []
        acc_gate,acc_ptr,acc_vac = 0.0, 0.0, 0.0
        # Run through decoder one time step at a time
        for t in range(self.max_r):
            decoder_ptr,decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
            all_decoder_outputs_vocab[t] = decoder_vacab
            topv, topvi = decoder_vacab.data.topk(1)
            all_decoder_outputs_ptr[t] = decoder_ptr
            topp, toppi = decoder_ptr.data.topk(1)
            top_ptr_i = torch.gather(input_batches[:,:,0],0,Variable(toppi.view(1, -1)))    
            next_in = [self._get_value(top_ptr_i.squeeze()[i]) if( toppi.squeeze()[i] < input_lengths[i]-1) else topvi.squeeze()[i] for i in range(batch_size)]

            decoder_input = Variable(torch.LongTensor(next_in)) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            temp = []
            from_which = []
            for i in range(batch_size):
                if(toppi.squeeze()[i] < len(p[i])-1 ):
                    temp.append(p[i][ toppi.squeeze()[i]])
                    from_which.append('p')
                else:
                    # this is for torch 0.3 ind = topvi.squeeze()[i]
                    ind = topvi.squeeze()[i].item()
                    if ind == EOS_token:
                        temp.append('<EOS>')
                    else:
                        temp.append(self.lang.index2word[ind])
                    from_which.append('v')
            decoded_words.append(temp)
            self.from_whichs.append(from_which)
        self.from_whichs = np.array(self.from_whichs)


        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        return decoded_words #, acc_ptr, acc_vac


    def evaluate(self,dev,avg_best,epoch,BLEU=False,Analyse=False,type='dev'):
        assert type=='dev' or type=='test'
        logging.info("STARTING EVALUATION:{}".format(type))
        acc_avg = 0.0
        wer_avg = 0.0
        bleu_avg = 0.0
        acc_P = 0.0
        acc_V = 0.0
        microF1_PRED,microF1_PRED_cal,microF1_PRED_nav,microF1_PRED_wet = [],[],[],[]
        microF1_TRUE,microF1_TRUE_cal,microF1_TRUE_nav,microF1_TRUE_wet = [],[],[],[]
        ref = []
        hyp = []
        ref_s = ""
        hyp_s = ""
        dialog_acc_dict = {}
        pbar = tqdm(enumerate(dev),total=len(dev))

        if Analyse == True:
            write_fp = write_to_disk('./mem-generate.txt')
            sum_copy_rate = []

        for j, data_dev in pbar: 
            if args['dataset']=='kvr':
                words = self.evaluate_batch(len(data_dev[1]),data_dev[0],data_dev[1],
                                    data_dev[2],data_dev[3],data_dev[4],data_dev[5],data_dev[6], data_dev[-2], data_dev[-1]) 
            else:
                words = self.evaluate_batch(len(data_dev[1]),data_dev[0],data_dev[1],
                        data_dev[2],data_dev[3],data_dev[4],data_dev[5],data_dev[6], data_dev[-4], data_dev[-3])          
            # acc_P += acc_ptr
            # acc_V += acc_vac
            acc=0
            w = 0 
            temp_gen = []



            for i, row in enumerate(np.transpose(words)):
                st = ''
                for e in row:
                    if e== '<EOS>': break
                    else: st+= e + ' '
                temp_gen.append(st)
                correct = data_dev[7][i]  
                ### compute F1 SCORE  
                if args['dataset']=='kvr':
                    f1_true,f1_pred = computeF1(data_dev[8][i],st.lstrip().rstrip(),correct.lstrip().rstrip())
                    microF1_TRUE += f1_true
                    microF1_PRED += f1_pred
                    f1_true,f1_pred = computeF1(data_dev[9][i],st.lstrip().rstrip(),correct.lstrip().rstrip())
                    microF1_TRUE_cal += f1_true
                    microF1_PRED_cal += f1_pred 
                    f1_true,f1_pred = computeF1(data_dev[10][i],st.lstrip().rstrip(),correct.lstrip().rstrip())
                    microF1_TRUE_nav += f1_true
                    microF1_PRED_nav += f1_pred 
                    f1_true,f1_pred = computeF1(data_dev[11][i],st.lstrip().rstrip(),correct.lstrip().rstrip()) 
                    microF1_TRUE_wet += f1_true
                    microF1_PRED_wet += f1_pred  
                elif args['dataset']=='babi' and int(self.task)==6:
                    f1_true,f1_pred = computeF1(data_dev[-2][i],st.lstrip().rstrip(),correct.lstrip().rstrip())
                    microF1_TRUE += f1_true
                    microF1_PRED += f1_pred

                if args['dataset']=='babi':
                    if data_dev[-1][i] not in dialog_acc_dict.keys():
                        dialog_acc_dict[data_dev[-1][i]] = []
                    if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                        acc+=1
                        dialog_acc_dict[data_dev[-1][i]].append(1)
                    else:
                        dialog_acc_dict[data_dev[-1][i]].append(0)
                else:
                    if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                        acc+=1
                #    print("Correct:"+str(correct.lstrip().rstrip()))
                #    print("\tPredict:"+str(st.lstrip().rstrip()))
                #    print("\tFrom:"+str(self.from_whichs[:,i]))

                # w += wer(correct.lstrip().rstrip(),st.lstrip().rstrip())
                ref.append(str(correct.lstrip().rstrip()))
                hyp.append(str(st.lstrip().rstrip()))
                ref_s+=str(correct.lstrip().rstrip())+ "\n"
                hyp_s+=str(st.lstrip().rstrip()) + "\n"

            # write batch data to disk
            if Analyse == True:
                for gen,gold in zip(temp_gen,data_dev[7]):
                    write_fp.write(gen+'\t' +gold +'\n')

                sum_copy_rate.append((np.array(self.from_whichs)=='p').sum() / (len(self.from_whichs)*len(self.from_whichs[0])))

            acc_avg += acc/float(len(data_dev[1]))
            wer_avg += w/float(len(data_dev[1]))            
            pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg/float(len(dev)),
                                                                    wer_avg/float(len(dev))))

        if Analyse == True:
            write_fp.close()


        # dialog accuracy
        if args['dataset']=='babi':
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k])==sum(dialog_acc_dict[k]):
                    dia_acc += 1
            logging.info("Dialog Accuracy:\t"+str(dia_acc*1.0/len(dialog_acc_dict.keys())))
            self._send_metrics(epoch,type, acc=dia_acc * 1.0 / len(dialog_acc_dict.keys()))
            self._send_metrics(epoch, type, acc_response=acc_avg / float(len(dev)))
        if args['dataset'] == 'kvr':
            f1 = f1_score(microF1_TRUE, microF1_PRED, average='micro')
            f1_cal = f1_score(microF1_TRUE_cal, microF1_PRED_cal, average='micro')
            f1_wet = f1_score(microF1_TRUE_wet, microF1_PRED_wet, average='micro')
            f1_nav = f1_score(microF1_TRUE_nav, microF1_PRED_nav, average='micro')
            logging.info("F1 SCORE:\t" + str(f1))
            logging.info("F1 CAL:\t" + str(f1_cal))
            logging.info("F1 WET:\t" + str(f1_wet))
            logging.info("F1 NAV:\t" + str(f1_nav))
            self._send_metrics(epoch,type, f1=f1, f1_cal=f1_cal, f1_wet=f1_wet, f1_nav=f1_nav)
        elif args['dataset']=='babi' and int(self.task)==6 :
            f1 = f1_score(microF1_TRUE, microF1_PRED, average='micro')
            logging.info("F1 SCORE:\t" + str(f1))
            self._send_metrics(epoch,type, babi_6_f1=f1)

              
        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True) 
        logging.info("BLEU SCORE:"+str(bleu_score))

        if Analyse ==False:
            self._send_metrics(epoch,type,total_loss=self.print_loss_avg,
                               ptr_loss=self.print_loss_ptr,
                               vocab_loss=self.print_loss_vac,
                               bleu_score=bleu_score)
        if (BLEU and bleu_score > 8):
            if (bleu_score >= avg_best):
                if type == 'dev':
                    directory=self.save_model(str(self.name)+str(bleu_score))
                    locals_var = locals()
                    self._save_runs_experiments(model_path=directory, epochs=epoch, bleu=bleu_score,
                                                f1=locals_var.get('f1', -1), f1_wet=locals_var.get('f1_wet', -1),
                                                f1_sch=locals_var.get('f1_cal', -1),
                                                f1_nav=locals_var.get('f1_nav', -1))
                    logging.info("MODEL SAVED")
            return bleu_score
        else:
            acc_avg = acc_avg/float(len(dev))
            if (acc_avg >= avg_best):
                if type == 'dev':
                    directory=self.save_model(str(self.name)+str(acc_avg))
                    locals_var = locals()
                    self._save_runs_experiments(model_path=directory, epochs=epoch, bleu=bleu_score,
                                                f1=locals_var.get('f1', -1), f1_wet=locals_var.get('f1_wet', -1),
                                                f1_sch=locals_var.get('f1_cal', -1),
                                                f1_nav=locals_var.get('f1_nav', -1))
                    logging.info("MODEL SAVED")
            return acc_avg

    def _send_metrics(self, epoch, type, **krgs):
        def show(win, epoch, value):
            opts = {}
            opts['title'] = win
            if self.vis.win_exists(win):
                self.vis.line(X=np.array([epoch]), Y=np.array([value]), win=win, update='append',opts=opts)
            else:
                self.vis.line(X=np.array([epoch]), Y=np.array([value]), win=win,opts=opts)
        if self.vis:
            for key,value in krgs.items():
                show(type+key,epoch,value)


    def _get_value(self,tensor_in):

        if hasattr(tensor_in,'item'):
            return tensor_in.item()
        else:
            return tensor_in.data[0]

def computeF1(entity,st,correct):
    y_pred = [0 for z in range(len(entity))]
    y_true = [1 for z in range(len(entity))]
    for k in st.lstrip().rstrip().split(' '):
        if (k in entity):
            y_pred[entity.index(k)] = 1
    return y_true,y_pred


class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
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
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        
    def get_state(self,bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(bsz, self.embedding_dim)).cuda()
        else:
            return Variable(torch.zeros(bsz, self.embedding_dim))


    def forward(self, story):
        story = story.transpose(0,1)
        story_size = story.size() # b * m * 3 
        if self.unk_mask:
            if(self.training):
                ones = np.ones((story_size[0],story_size[1],story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))],1-self.dropout)[0]
                ones[:,:,0] = ones[:,:,0] * rand_mask
                a = Variable(torch.Tensor(ones))
                if USE_CUDA: a = a.cuda()
                story = story*a.long()
        u = [self.get_state(story.size(0))]
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1).long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size+(embed_A.size(-1),)) # b * m * s * e
            m_A = torch.sum(embed_A, 2).squeeze(2) # b * m * e

            # 每个词有三个属性,【词，说话者,第几轮说的】相当于将一个词又拓展成三个词了

            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob   = self.softmax(torch.sum(m_A*u_temp, 2))  
            embed_C = self.C[hop+1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),)) 
            m_C = torch.sum(embed_C, 2).squeeze(2)

            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)   
        return u_k

class DecoderrMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
        super(DecoderrMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.W = nn.Linear(embedding_dim,1)
        self.W1 = nn.Linear(2*embedding_dim,self.num_vocab)
        self.gru = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)

    def load_memory(self, story):
        story_size = story.size() # b * m * 3 
        if self.unk_mask:
            if(self.training):
                ones = np.ones((story_size[0],story_size[1],story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))],1-self.dropout)[0]
                ones[:,:,0] = ones[:,:,0] * rand_mask
                a = Variable(torch.Tensor(ones))
                if USE_CUDA:
                    a = a.cuda()
                story = story*a.long()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1))#.long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size+(embed_A.size(-1),)) # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2) # b * m * e
            m_A = embed_A    
            embed_C = self.C[hop+1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),)) 
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            m_C = embed_C
            self.m_story.append(m_A)
        self.m_story.append(m_C)

    def ptrMemDecoder(self, enc_query, last_hidden):
        embed_q = self.C[0](enc_query) # b * e
        output, hidden = self.gru(embed_q.unsqueeze(0), last_hidden)
        temp = []
        u = [hidden[0].squeeze()]   
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if(len(list(u[-1].size()))==1): u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_lg = torch.sum(m_A*u_temp, 2)
            prob_   = self.softmax(prob_lg)
            m_C = self.m_story[hop+1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            if (hop==0):
                p_vocab = self.W1(torch.cat((u[0], o_k),1))
            u_k = u[-1] + o_k
            u.append(u_k)
        p_ptr = prob_lg 
        return p_ptr, p_vocab, hidden




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
