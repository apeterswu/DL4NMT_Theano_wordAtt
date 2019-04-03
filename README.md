# DL4NMT_Theano with word Attention

AAAI2018:
```
@inproceedings{wu2018word,
  title={Word attention for sequence to sequence text understanding},
  author={Wu, Lijun and Tian, Fei and Zhao, Li and Lai, Jianhuang and Liu, Tie-Yan},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}
```

Deep neural machine translation (NMT) model, implemented in Theano.

## Install in new node

```bash

git clone https://github.com/apeterswu/DL4NMT_Theano_wordAtt.git
git clone https://github.com/Lasagne/Lasagne.git
cd Lasagne
pip install .
cd ../DL4NMT_Theano_wordAtt
mkdir -p log/complete translated/complete model/complete
#copy data from other nodes to here...
```

**NOTE**：在node（包括GCR）上跑job之前，请确保code是最新的，在Project根目录下运行`git pull`。

## Train

Run `train_nmt.py`.

See `train_nmt.py -h` for help.


**NOTE**：由于shuffle data per epoch的存在，当一个job使用的dataset没有shuffle版本（下标为0,1,2,...）时，会立即创建一个。
因此**不要**同时交两个dataset没有shuffle版本的job，防止冲突。等一个把下标为0的shuffle版本创建出来之后再交另外的。

## Options

所有options见`config.py`。

可配置的options见`train_nmt.py`。

## Dataset

dataset由`--dataset` option控制，所有dataset见`constants.py`。

dataset由training data，small training data，validation data，dictionary组成。


### 对dataset进行Truecase转换

运行`make_truecase.bat`脚本，要求原始dataset分别在各自目录下（`data/train, data/test, data/dev, data/dic`）。

该脚本具体用法见其中Usage行，最后一个可选项为single_dict，若设置，则source和target使用同一个dict，名字为single_dict。


## Test

Test脚本以"test_"开头。


以_single结尾的脚本用于linux server，其余用于windows server。

BPE test脚本在Windows上需要用bash运行，gdw135和gdw144均安装了bash。

脚本用法可见脚本里面的Usage行。

truecase的test应该在运行`perl multi-bleu.perl`之前运行`perl detruecase.perl < translated_file > output`。


## NOTES

Scripts in `scripts` must be call at root directory of the project (the directory of this README).
