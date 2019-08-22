import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils.masked_cross_entropy import *
from utils.config import *
import random
import numpy as np
import datetime
from utils.measures import wer,moses_multi_bleu
from sklearn.metrics import f1_score
import math
from visdom import Visdom
import warnings
import codecs

class write_to_disk():
    def __init__(self,path):
        self.fp= codecs.open(path, 'w', 'utf8')
        print('result writing to ',path)
    def write(self,sentence):
        self.fp.write(sentence)
    def close(self):
        self.fp.close()


class LuongSeqToSeq(nn.Module):
    def __init__(self,embedding_size,hidden_size,max_len,max_r,lang,path,task,lr=0.01,n_layers=1, dropout=0.1,env_name=None,
                 debug_host_ip=None):
        super(LuongSeqToSeq, self).__init__()
        assert  env_name != None
        assert debug_host_ip!=None
        if embedding_size == None:
            warnings.warn('Word embedding size are none,set to be same with hiddensize:{}'.format(hidden_size))
            embedding_size = hidden_size
        self.name = "LuongSeqToSeq"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.max_len = max_len ## max input
        self.max_r = max_r ## max responce len    
        self.lang = lang
        self.lr = lr
        self.decoder_learning_ratio = 5.0
        self.n_layers = n_layers
        self.dropout = dropout
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
            self.encoder = EncoderRNN(input_size=lang.n_words,
                                      word_size=embedding_size,
                                      hidden_size=hidden_size,
                                      n_layers=n_layers,
                                      dropout=dropout)
            self.decoder = LuongAttnDecoderRNN(hidden_size=hidden_size,
                                               word_size=embedding_size,
                                               output_size=lang.n_words, n_layers=n_layers, dropout=dropout)
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr * self.decoder_learning_ratio)

        self.loss = 0
        self.loss_vac = 0  
        self.print_every = 1
        self.print_loss_avg = 0
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()
        try:
            self.vis = Visdom(server=debug_host_ip,env=env_name)
            print('server ip:{},env_name:{}'.format(debug_host_ip,env_name))
        except:
            print("Can't not use visdom ")
            self.vis=None
    def print_loss(self): 
        print_loss_avg = self.loss / self.print_every
        self.print_loss_avg = print_loss_avg
        self.print_every += 1
        return 'L:{:.2f}'.format(print_loss_avg)
    

    def save_model(self,dec_type):
        name_data = "KVR/" if self.task=='' else "BABI/"
        if USEKB:
            directory = 'save/Luong_KB-'+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'DR'+str(self.dropout)+'L'+str(self.n_layers)+'lr'+str(self.lr)+str(dec_type)         
        else:
            directory = 'save/Luong_noKB-'+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'DR'+str(self.dropout)+'L'+str(self.n_layers)+'lr'+str(self.lr)+str(dec_type)         
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory+'/enc.th')
        torch.save(self.decoder, directory+'/dec.th')
        
    def load_model(self,file_name_enc,file_name_dec):
        self.encoder = torch.load(file_name_enc)
        self.decoder = torch.load(file_name_dec)


    def train_batch(self, input_batches, input_lengths, target_batches, 
                    target_lengths, target_index, target_gate, batch_size, clip,
                    teacher_forcing_ratio, reset):    
        if reset:
            self.loss = 0
            self.loss_vac = 0  
            self.print_every = 1
            
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss_Vocab,loss_Ptr,loss_Gate = 0,0,0
        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths)
      
        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers],encoder_hidden[1][:self.decoder.n_layers])

        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = Variable(torch.zeros(max_target_length, batch_size, self.output_size))
        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            decoder_input = decoder_input.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        if use_teacher_forcing:    
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_vacab, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

                all_decoder_outputs_vocab[t] = decoder_vacab
                decoder_input = target_batches[t] # Next input is current target
                if USE_CUDA: decoder_input = decoder_input.cuda()
                
        else:
            for t in range(max_target_length):
                decoder_vacab,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                all_decoder_outputs_vocab[t] = decoder_vacab
                topv, topi = decoder_vacab.data.topk(1)
                decoder_input = Variable(topi.view(-1)) # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()
                  
        #Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths
        )

        loss = loss_Vocab
        loss.backward()
        
        # Clip gradient norms
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()

    def evaluate_batch(self,batch_size,input_batches, input_lengths, target_batches):  
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)  
        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)
        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers],encoder_hidden[1][:self.decoder.n_layers])

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(self.max_r, batch_size, self.decoder.output_size))
        # Move new Variables to CUDA

        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            decoder_input = decoder_input.cuda()
        
        # Run through decoder one time step at a time
        for t in range(self.max_r):
            decoder_vacab,decoder_hidden  = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            all_decoder_outputs_vocab[t] = decoder_vacab
            topv, topi = decoder_vacab.data.topk(1)
            decoder_input = Variable(topi.view(-1))
    
            decoded_words.append(['<EOS>'if ni == EOS_token else self.lang.index2word[ni.item()] for ni in topi.view(-1)])
            # Next input is chosen word
            if USE_CUDA: decoder_input = decoder_input.cuda()
        
        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        
        return decoded_words


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

        if Analyse == True:
            write_fp = write_to_disk('./luoattn-seq-generate.txt')

        pbar = tqdm(enumerate(dev),total=len(dev))
        for j, data_dev in pbar: 
            words = self.evaluate_batch(len(data_dev[1]),data_dev[0],data_dev[1],data_dev[2])             
            acc=0
            w = 0
            temp_gen = []
            for i, row in enumerate(np.transpose(words)):
                st = ''
                for e in row:
                    if e== '<EOS>':
                        break
                    else:
                        st+= e + ' '
                temp_gen.append(st)
                correct = data_dev[7][i]  
                ### compute F1 SCORE  
                if args['dataset'] == 'kvr':
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
                elif args['dataset'] == 'babi' and int(self.task) == 6:
                    f1_true, f1_pred = computeF1(data_dev[-2][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE += f1_true
                    microF1_PRED += f1_pred

                if args['dataset']=='babi':
                    if data_dev[-1][i] not in dialog_acc_dict.keys():
                        dialog_acc_dict[data_dev[-1][i]] = []
                    if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                        acc += 1
                        dialog_acc_dict[data_dev[-1][i]].append(1)
                    else:
                        dialog_acc_dict[data_dev[-1][i]].append(0)
                else:
                    if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                        acc += 1
                # else:
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
                
            acc_avg += acc/float(len(data_dev[1]))
            wer_avg += w/float(len(data_dev[1]))
            pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg/float(len(dev)),
                                                                    wer_avg/float(len(dev))))
        if Analyse == True:
            write_fp.close()

        if args['dataset'] == 'babi':
            # TODO:计算平均的对话准确度
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            logging.info("Dialog Accuracy:\t" + str(dia_acc * 1.0 / len(dialog_acc_dict.keys())))
            self._send_metrics(epoch, type,acc=dia_acc * 1.0 / len(dialog_acc_dict.keys()))
            self._send_metrics(epoch, type,acc_response=acc_avg/ float(len(dev)))

        if args['dataset'] == 'kvr':
            f1 = f1_score(microF1_TRUE, microF1_PRED, average='micro')
            f1_cal = f1_score(microF1_TRUE_cal, microF1_PRED_cal, average='micro')
            f1_wet = f1_score(microF1_TRUE_wet, microF1_PRED_wet, average='micro')
            f1_nav = f1_score(microF1_TRUE_nav, microF1_PRED_nav, average='micro')

            logging.info("F1 SCORE:\t"+str(f1))
            logging.info("F1 CAL:\t"+str(f1_cal))
            logging.info("F1 WET:\t"+str(f1_wet))
            logging.info("F1 NAV:\t"+str(f1_nav))
            self._send_metrics(epoch,type, f1=f1, f1_cal=f1_cal, f1_wet=f1_wet, f1_nav=f1_nav)

        elif args['dataset'] =='babi' and int(self.task) ==6:
            f1 = f1_score(microF1_TRUE, microF1_PRED, average='micro')
            logging.info("F1 SCORE:\t" + str(f1))
            self._send_metrics(epoch,type, babi_6_f1=f1)

        self._send_metrics(epoch,type,total_loss=self.print_loss_avg)
        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        logging.info("BLEU SCORE:" + str(bleu_score))
        self._send_metrics(epoch, type, bleu=bleu_score)

        if (BLEU):
            if (bleu_score >= avg_best):
                if type == 'dev':
                    self.save_model(str(self.name)+str(bleu_score))
                    logging.info("MODEL SAVED")
            return bleu_score
        else:
            acc_avg = acc_avg/float(len(dev))
            if (acc_avg >= avg_best):
                if type == 'dev':
                    self.save_model(str(self.name)+str(acc_avg))
                    logging.info("MODEL SAVED")
            return acc_avg

    def _send_metrics(self, epoch,type, **krgs):
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

def computeF1(entity,st,correct):
    y_pred = [0 for z in range(len(entity))]
    y_true = [1 for z in range(len(entity))]
    for k in st.lstrip().rstrip().split(' '):
        if (k in entity):
            y_pred[entity.index(k)] = 1
    return y_true,y_pred

class EncoderRNN(nn.Module):
    def __init__(self, input_size,word_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()      
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout       
        self.embedding = nn.Embedding(input_size, word_size)
        self.embedding_dropout = nn.Dropout(dropout) 
        self.lstm = nn.LSTM(word_size, hidden_size, n_layers, dropout=self.dropout)
        if USE_CUDA:
            self.lstm = self.lstm.cuda() 
            self.embedding_dropout = self.embedding_dropout.cuda()
            self.embedding = self.embedding.cuda() 

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(1)
        c0_encoder = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))  
        h0_encoder = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)) ### * self.num_directions = 2 if bi
        if USE_CUDA:
            h0_encoder = h0_encoder.cuda()
            c0_encoder = c0_encoder.cuda() 
        return (h0_encoder, c0_encoder)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.embedding_dropout(embedded)
        hidden = self.get_state(input_seqs)
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)    
        outputs, hidden = self.lstm(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        return outputs, hidden


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size,word_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.word_size=word_size

        # Define layers
        self.embedding = nn.Embedding(output_size, word_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(word_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.W1 = nn.Linear(2*hidden_size, hidden_size)
        # self.v = nn.Linear(hidden_size, 1)
        self.v = nn.Parameter(torch.rand(hidden_size)).cuda()
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)   
        if USE_CUDA:
            self.embedding = self.embedding.cuda()
            self.dropout = self.dropout.cuda()
            self.lstm = self.lstm.cuda()
            self.out = self.out.cuda() 
            self.W1 = self.W1.cuda()
            self.concat = self.concat.cuda()

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        max_len = encoder_outputs.size(0)
        encoder_outputs = encoder_outputs.transpose(0,1)
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)
        embedded = embedded.view(1, batch_size, self.word_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.lstm(embedded, last_hidden)

        s_t = hidden[0][-1].unsqueeze(0)
        H = s_t.repeat(max_len,1,1).transpose(0,1)

        energy = F.tanh(self.W1(torch.cat([H,encoder_outputs], 2)))
        energy = energy.transpose(2,1)
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        a = F.softmax(energy)
        context = a.bmm(encoder_outputs)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden
