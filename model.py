#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import OrderedDict
import sys
import os

import theano
import theano.tensor as T
import numpy as np
import cPickle as pkl

from constants import fX, profile
from utils import *
from layers import *

__author__ = 'fyabc'


class ParameterInitializer(object):
    def __init__(self, options):
        # Dict of options
        self.O = options

    @staticmethod
    def init_embedding(np_parameters, name, n_in, n_out):
        np_parameters[name] = normal_weight(n_in, n_out)

    def init_input_to_context(self, parameters, reload_=None, preload=None,
                              load_embedding=True):
        """Initialize the model parameters from input to context vector.

        :param parameters: OrderedDict of Theano shared variables to be initialized.
        """

        np_parameters = OrderedDict()

        # Source embedding
        self.init_embedding(np_parameters, 'Wemb', self.O['n_words_src'], self.O['dim_word'])

        # Encoder: bidirectional RNN
        for layer_id in xrange(self.O['n_encoder_layers']):
            if layer_id == 0:
                n_in = self.O['dim_word']
            else:
                n_in = self.O['dim']
            np_parameters = get_init(self.O['encoder'])(self.O, np_parameters, prefix='encoder', nin=n_in,
                                                        dim=self.O['dim'], layer_id=layer_id)
            np_parameters = get_init(self.O['encoder'])(self.O, np_parameters, prefix='encoder_r', nin=n_in,
                                                        dim=self.O['dim'], layer_id=layer_id)

        # Reload parameters
        reload_ = self.O['reload_'] if reload_ is None else reload_
        preload = self.O['preload'] if preload is None else preload
        if reload_ and os.path.exists(preload):
            print('Reloading model parameters')
            np_parameters = load_params(preload, np_parameters)
        else:
            if load_embedding:
                # [NOTE] Important: Load embedding even in random init case
                print('Loading embedding')
                old_params = np.load(self.O['preload'])
                np_parameters['Wemb'] = old_params['Wemb']

        print_params(np_parameters)

        # Init theano parameters
        init_tparams(np_parameters, parameters)


class NMTModel(object):
    """The model class.

    This is a light weight class, just contains some needed elements and model components.
    The main work is done by the caller.
    """

    # This is a simple wrapper of layers.py now.
    # todo: Move code from layers.py to here.

    def __init__(self, options):
        # Dict of options
        self.O = options

        # Dict of parameters (Theano shared variables)
        self.P = OrderedDict()

        # Instance of ParameterInitializer, init the parameters.
        self.initializer = ParameterInitializer(options)

    # Methods to build the each component of the model

    @staticmethod
    def get_input():
        """Get model input.

        Model input shape: #words * #samples

        :return: 4 Theano variables:
            x, x_mask, y, y_mask
        """

        x = T.matrix('x', dtype='int64')
        x_mask = T.matrix('x_mask', dtype=fX)
        y = T.matrix('y', dtype='int64')
        y_mask = T.matrix('y_mask', dtype=fX)

        return x, x_mask, y, y_mask

    @staticmethod
    def input_dimensions(x, y):
        """Get input dimensions.

        :param x: input x
        :param y: input y
        :return: 3 Theano variables:
            n_timestep, n_timestep_tgt, n_samples
        """

        n_timestep = x.shape[0]
        n_timestep_tgt = y.shape[0]
        n_samples = x.shape[1]

        return n_timestep, n_timestep_tgt, n_samples

    @staticmethod
    def reverse_input(x, x_mask):
        return x[::-1], x_mask[::-1]

    def embedding(self, input_, n_timestep, n_samples, emb_name='Wemb'):
        """Embedding layer: input -> embedding"""

        return embedding(self.P, input_, self.O, n_timestep, n_samples, emb_name)

    def feed_forward(self, input_, prefix, activation=tanh):
        """Feed-forward layer."""

        return fflayer(self.P, input_, self.O, prefix, activation)

    def gru_encoder(self, src_embedding, src_embedding_r, x_mask, xr_mask, dropout_params=None):
        """GRU encoder layer: source embedding -> encoder context"""

        return gru_encoder(self.P, src_embedding, src_embedding_r, x_mask, xr_mask, self.O, dropout_params)

    @staticmethod
    def get_context_mean(context, x_mask):
        """Get mean of context (across time) as initial state of decoder RNN
        
        Or you can use the last state of forward + backward encoder RNNs
            # return concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)
        """

        return (context * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    def input_to_context(self, given_input=None):
        """Build the part of the model that from input to context vector.

        Used for regression of deeper encoder.

        :param given_input: List of input Theano tensors or None
            If None, this method will create them by itself.
        :returns tuple of input list and output
        """

        x, x_mask, y, y_mask = self.get_input() if given_input is None else given_input
        x_r, x_mask_r = self.reverse_input(x, x_mask)

        n_timestep, n_timestep_tgt, n_samples = self.input_dimensions(x, y)

        # Word embedding for forward rnn and backward rnn (source)
        src_embedding = self.embedding(x, n_timestep, n_samples)
        src_embedding_r = self.embedding(x_r, n_timestep, n_samples)

        context = self.gru_encoder(src_embedding, src_embedding_r, x_mask, x_mask_r, dropout_params=None)

        return [x, x_mask, y, y_mask], context

    def input_to_decoder_context(self, given_input=None):
        """Build the part of the model that from input to context vector of decoder.
        
        :param given_input: List of input Theano tensors or None
            If None, this method will create them by itself.
        :return: tuple of input list and output
        """

        (x, x_mask, y, y_mask), context = self.input_to_context(given_input)

        n_timestep, n_timestep_tgt, n_samples = self.input_dimensions(x, y)

        context_mean = self.get_context_mean(context, x_mask)
        # Initial decoder state
        init_decoder_state = self.feed_forward(context_mean, prefix='ff_state', activation=tanh)

        # Word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.

        # todo

    def save_whole_model(self, model_file, iteration):
        # save with iteration
        save_filename = '{}_iter{}.iter160000.npz'.format(
            os.path.splitext(model_file)[0], iteration,
        )

        print('Saving the new model at iteration {} to {}...'.format(iteration, save_filename), end='')

        # Encoder weights from new model + other weights from old model
        old_params = dict(np.load(self.O['preload']))
        old_params.update(unzip(self.P))

        np.savez(save_filename, **old_params)

        # Save options
        with open('{}.pkl'.format(save_filename), 'wb') as f:
            pkl.dump(self.O, f)

        print('Done')
        sys.stdout.flush()
