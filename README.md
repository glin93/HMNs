# Introduction
Thanks to the great work in Mem2seq, we code is built upon [Mem2seq](https://github.com/HLTCHKUST/Mem2Seq). bAbI tasks can be found [here](https://research.fb.com/downloads/babi/) and DSTC 2 data can be found [here](http://camdial.org/~mh521/dstc/).

# Requirements
```bash
pip install requirements.txt
```
Change permision for BLEU evaluation script
```bash
chmod +777 tools/multi-bleu.perl
```
# Commands to Reproduce HMNs Results HMNs

If you want to run on GPUs, please modify the variable USE_CUDA(in utils/config.py) to be Ture, or set it to be False when you want to run it on CPU.

1. bAbI-3

```bash
python multi_mem_train.py -lr=0.001 -layer=1 -hdd=256 -dr=0.1 -dec=Mem2Seq -bsz=64 -ds=babi -t=3 -kb-layer=1 -enbirnn -debirnn
```

2. bAbI-4
```bash
python multi_mem_train.py -lr=0.001 -layer=1 -hdd=256 -dr=0.2 -dec=Mem2Seq -bsz=64 -ds=babi -t=4 -kb-layer=1 -enbirnn -debirnn
```

3. bAbI-5
```bash
python multi_mem_train.py -lr=0.001 -layer=1 -hdd=256 -dr=0.1 -dec=LuongSeqToSeq -bsz=64 -ds=babi -t=5  -kb-layer=1 -enbirnn -debirnn
```

4. DSTC2
```bash
python multi_mem_train.py -lr=0.001 -layer=1 -hdd=128 -dr=0.1 -dec=Mem2Seq -bsz=64 -ds=babi -t=6 -kb-layer=1 -enbirnn -debirnn  -load-limits=15 
```

5. Kek Value Dataset
```bash
python multi_mem_train.py -lr=0.001 -layer=3 -hdd=256 -dr=0.1 -dec=Mem2Seq -bsz=64 -ds=kvr -t= -kb-layer=3 -enbirnn -debirnn 
```