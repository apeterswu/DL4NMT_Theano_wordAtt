from __future__ import print_function

import os
import cPickle as pkl
from pprint import pprint
import re
import errno
import random
import gzip
import sys
import warnings
import os

import theano
import theano.tensor as tensor
from libs.models import NMTModel
from libs.utility.utils import *
import numpy as np
from libs.config import DefaultOptions
from libs.models import build_and_init_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model_prefix', help='model file prefix.')
parser.add_argument('start', type=int, help='start iteration number.')
parser.add_argument('end', type=int, help='end iteration number')
parser.add_argument('gap', type=int, default=10000, help='the gap between each saved model.')

args = parser.parse_args()

num_of_model = args.end - args.start + 1

option_file = '%s.iter%d.npz.pkl' % (os.path.splitext(args.model_prefix)[0], args.start * args.gap)
with open(option_file, 'rb') as f:
    options = DefaultOptions.copy()
    options.update(pkl.load(f))

    if 'fix_dp_bug' not in options:
        options['fix_dp_bug'] = False

    # pprint(options)

model = NMTModel(options)

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
trng = RandomStreams(1234)
use_noise = theano.shared(np.float32(0.))

print('initialize the model.')
params = model.initializer.init_params()
print('Done')
trans_model_file = '%s.iter%d.npz' % (os.path.splitext(args.model_prefix)[0], args.start * args.gap)
old_params = np.load(trans_model_file)
for key, value in old_params.iteritems():
    if key not in old_params:
        warnings.warn('{} is not in the archive'.format(key))
        continue
    if params[key].shape == old_params[key].shape:
        params[key] = old_params[key]

for idx in xrange(args.start + 1, args.end + 1):
    print('load model file.')
    trans_model_file = '%s.iter%d.npz' % (os.path.splitext(args.model_prefix)[0], idx * args.gap)
    old_params = np.load(trans_model_file)
    for key, value in old_params.iteritems():
        if key not in old_params:
            warnings.warn('{} is not in the archive'.format(key))
            continue
        if params[key].shape == old_params[key].shape:
            params[key] += old_params[key]
    print('add one model values.')
for key, value in params.iteritems():
    params[key] = value / num_of_model
sys.stdout.flush()

model.init_tparams(params)
history = []
uidx = 123
print('Save the averaged model. The uidx = 123...')
model.save_model(args.model_prefix, history, uidx)
print('Done')
sys.stdout.flush()

# def load_params(path, params):
#     """Load parameters
#
#     :param path: Path of old parameters.
#     :param params: New parameters to be updated.
#     """
#
#     old_params = np.load(path)
#     for key, value in params.iteritems():
#         if key not in old_params:
#             warnings.warn('{} is not in the archive'.format(key))
#             continue
#         if params[key].shape == old_params[key].shape:
#             params[key] = old_params[key]
#
#     return params

