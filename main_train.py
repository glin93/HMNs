import numpy as np
import logging
from tqdm import tqdm

from utils.config import *
from models.enc_vanilla import *
from models.enc_Luong import *
from models.enc_PTRUNK import *
from models.Mem2Seq import *

BLEU = False

if (args['decoder'] == "Mem2Seq"):
    if args['dataset'] == 'kvr':
        from utils.utils_kvr_mem2seq import *

        BLEU = True
    elif args['dataset'] == 'babi':
        from utils.utils_babi_mem2seq import *
    else:
        print("You need to provide the --dataset information")
else:
    if args['dataset'] == 'kvr':
        from utils.utils_kvr import *

        BLEU = True
    elif args['dataset'] == 'babi':
        from utils.utils_babi import *
    else:
        print("You need to provide the --dataset information")

# Configure models
avg_best, cnt, acc = 0.0, 0, 0.0
cnt_1 = 0
### LOAD DATA
train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'], batch_size=int(args['batch']),
                                                                   shuffle=True)

if args['decoder'] == "Mem2Seq":
    model = globals()[args['decoder']](hidden_size=int(args['hidden']),
                                       max_len=max_len,
                                       max_r=max_r,
                                       lang=lang,
                                       path=args['path'],
                                       task=args['task'],
                                       lr=float(args['learn']),
                                       n_layers=int(args['layer']),
                                       dropout=float(args['drop']),
                                       unk_mask=bool(int(args['unk_mask'])),
                                       env_name='singlemem' + args['dataset'] + args['task'] + 'hop' + str(
                                           args['layer']) + '_dr{}_hdd{}'.format(args['drop'], args['hidden']) +args['addition_name'],
                                       debug_host_ip=args['serverip']
                                       )
else:
    model = globals()[args['decoder']](
                                    embedding_size=int(args['embeddingsize']),
                                    hidden_size=int(args['hidden']),
                                    max_len=max_len,
                                    max_r= max_r,
                                    lang=lang,
                                    path=args['path'],
                                    task=args['task'],
                                    lr=float(args['learn']),
                                    n_layers=int(args['layer']),
                                    dropout=float(args['drop']),
                                    env_name=args['decoder'] + args['dataset'] + args['task'] + 'hop' + str(
                                           args['layer']) + '_dr{}_hdd{}'.format(args['drop'], args['hidden']),
                                    debug_host_ip=args['serverip']
                                )

for epoch in range(int(args['epoch'])):
    logging.info("Epoch:{}".format(epoch))
    # Run the train function
    pbar = tqdm(enumerate(train), total=len(train))
    for i, data in pbar:
        if args['decoder'] == "Mem2Seq":
            if args['dataset'] == 'kvr':
                model.train_batch(data[0], data[1], data[2], data[3], data[4], data[5],
                                  len(data[1]), 10.0, 0.5, data[-2], data[-1], i == 0)
            else:
                model.train_batch(data[0], data[1], data[2], data[3], data[4], data[5],
                                  len(data[1]), 10.0, 0.5, data[-4], data[-3], i == 0)
        else:
            model.train_batch(data[0], data[1], data[2], data[3], data[4], data[5],
                              len(data[1]), 10.0, 0.5, i == 0)
        pbar.set_description(model.print_loss())

    if ((epoch + 1) % int(args['evalp']) == 0):
        acc = model.evaluate(dev, avg_best, epoch, BLEU,type='dev')
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
        # if(cnt == 5): break
        if (acc == 1.0): break
