import numpy as np
import logging
from tqdm import tqdm
import sys
from utils.config import *

# import data
if args['dataset'] == 'kvr':
    from utils.utils_kvr_multi_mem import *

    BLEU = True
elif args['dataset'] == 'babi':
    from utils.utils_babi_multi_mem import *
    BLEU =False
else:
    print("You need to provide the --dataset information")


# Configure models
avg_best, cnt, acc = 0.0, 0, 0.0
cnt_1 = 0



if  args['model_name'] == 'rnn_mem':

    print('Using rnn in memory')
    from multi_mem_base_rnn_mem.multi_memory import Mem2Seq

    train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'], batch_size=int(args['batch']))

    model = Mem2Seq(int(args['hidden']), max_len, max_r, lang, args['path'], args['task'],
                    lr=float(args['learn']),
                    n_layers=int(args['layer']),
                    dropout=float(args['drop']),
                    unk_mask=bool(int(args['unk_mask'])),
                    kb_mem_hop=int(args['kb_layer']),
                    env_name='multi'+args['dataset']+args['task']+'hop' + str(args['layer'])+'KB'+args['kb_layer']+'_limits{}'.format(args['load_limits']) +'_dr{}hdd{}'.format(args['drop'],args['hidden']) +args['addition_name'],
                    debug_host_ip=args['serverip'],
                    debirnn=args['debirnn'],
                    enbirnn=args['enbirnn']
                    )
else:
    raise RuntimeError('No defined parameters')


# 下面开始训练过程,dynamic的数据读取有些不一样,所以需要判断一下

print('Using bleu:{}'.format(BLEU))
for epoch in range(int(args['epoch'])):
    logging.info("Epoch:{}".format(epoch))
    # Run the train function
    pbar = tqdm(enumerate(train), total=len(train))

    for i, data in pbar:

        model.train_batch(
            input_batches=data[0],
            input_lengths=data[1],
            target_batches=data[2],
            target_lengths=data[3],
            target_index=data[4],
            target_gate=data[5],
            batch_size=len(data[1]),
            clip=10,
            teacher_forcing_ratio=0.5,
            conv_seqs=data[-6],
            conv_lengths=data[-5],
            kb_seqs=data[-4],
            kb_lengths=data[-3],
            kb_target_index=data[-2],
            kb_plain=data[-1],
            reset=i == 0
        )

        pbar.set_description(model.print_loss())


    if ((epoch + 1) % int(args['evalp']) == 0):
        acc = model.evaluate(dev, avg_best,epoch=epoch, BLEU= BLEU,type='dev')
        if args['dataset']=='babi' and int(args['task'])<=5:
            # test oov datasets
            print('testing on oov')
            model.evaluate(testOOV,avg_best,epoch=epoch,BLEU=BLEU,type='test')
        if 'Mem2Seq' in args['decoder']:
            model.scheduler.step(acc)
        if (acc >= avg_best):
            avg_best = acc
            cnt = 0
        else:
            cnt += 1
        # if (cnt == 5): break
        if (acc == 1.0): break



