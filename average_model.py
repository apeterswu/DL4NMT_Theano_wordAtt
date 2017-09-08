from __future__ import print_function

import os
import cPickle as pkl
from pprint import pprint
import re
import errno
import random
import gzip
import sys
import time

import theano
import theano.tensor as tensor
from libs.models.model import *
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

    pprint(options)

model = NMTModel(options)
params = model.initializer.init_params()

for idx in xrange(args.start, args.end + 1):
    trans_model_file = '%s.iter%d.npz' % (os.path.splitext(args.model_prefix)[0], idx * args.gap)
    old_params = np.load(trans_model_file)
    for key, value in old_params.iteritems():
        params[key] += value
for key in params.keys():
    params[key] /= (num_of_model * 1.0)

model.init_tparams(params)

print('Save the averaged model. The uidx = 0...')
model.save_model(args.model_prefix, None, 0)
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

