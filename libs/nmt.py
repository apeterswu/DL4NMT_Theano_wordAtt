"""
Build a neural machine translation model with soft attention
"""

import cPickle as pkl
import copy
import os
import sys
import time
import math
from pprint import pprint

import numpy as np
import theano
import theano.tensor as tensor

from .constants import profile, fX
from .utility.data_iterator import TextIterator, SequentialTextIterator
from .utility.optimizers import Optimizers
from .utility.utils import *

from .utility.translate import translate_dev_get_bleu
from .models import NMTModel, TrgAttnNMTModel


def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, normalize=False):
    """Calculate the log probablities on a given corpus using translation model"""

    probs = []

    n_done = 0

    for x, y in iterator:
        n_done += len(x)

        lengths = np.array([len(s) for s in x])

        x, x_mask, y, y_mask = prepare_data(x, y)

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        if normalize:
            pprobs = pprobs / lengths

        for pp in pprobs:
            probs.append(pp)

        sys.stdout.write('\rDid ' + str(n_done) + ' samples')

    print
    return np.array(probs)


def validation(iterator, f_cost):
    valid_cost = 0.0
    valid_count = 0
    for x, y in iterator:
        x, x_mask, y, y_mask = prepare_data(x, y, maxlen=None)

        if x is None:
            continue

        valid_cost += f_cost(x, x_mask, y, y_mask) * x_mask.shape[1]
        valid_count += x_mask.shape[1]

    return valid_cost / valid_count


def visual_test_attention(iterator, f_att, f_att_gate):
    alpha_attention_file = open('alpha_attention_score.txt', 'w')
    beta_attention_file = open('beta_attention_score.txt', 'w')
    gated_attention_file = open('gated_attention_score.txt', 'w')

    # only one sentence in test_iterator batch
    for seq_x, seq_y, origin_seqx, origin_seqy in iterator:
        x, x_mask, y, y_mask = prepare_data(seq_x, seq_y, maxlen=None)
        if x is None:
            continue
        alpha, beta = f_att(x, x_mask, y, y_mask)
        average_attention_gate = f_att_gate(x, x_mask, y, y_mask)
        alpha_attention_file.write(origin_seqx[0])
        alpha_attention_file.write(origin_seqy[0])
        beta_attention_file.write(origin_seqx[0])
        beta_attention_file.write(origin_seqy[0])
        gated_attention_file.write(origin_seqx[0])
        gated_attention_file.write(origin_seqy[0])
        # [target_sen * batch * source_sen]
        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[2]):
                alpha_attention_file.write(str(alpha[i][0][j]) + ' ')
                beta_attention_file.write(str(beta[i][0][j]) + ' ')
                gated_attention = alpha[i][0][j] * average_attention_gate[i][0] + beta[i][0][j] * \
                                                                                  (1 - average_attention_gate[i][0])
                gated_attention_file.write(str(gated_attention) + ' ')
            alpha_attention_file.write('\n')
            beta_attention_file.write('\n')
            gated_attention_file.write('\n')
    alpha_attention_file.close()
    beta_attention_file.close()
    gated_attention_file.close()


def visual_test_attention_gate(iterator, f_att_gate):
    attention_gate_file = open('attention_gate_score.txt', 'w')

    # only one sentence in test_iterator batch
    for seq_x, seq_y, origin_seqx, origin_seqy in iterator:
        x, x_mask, y, y_mask = prepare_data(seq_x, seq_y, maxlen=None)
        if x is None:
            continue
        average_attention_gate = f_att_gate(x, x_mask, y, y_mask)
        attention_gate_file.write(origin_seqx[0])
        attention_gate_file.write(origin_seqy[0])
        # [target_sen * batch * dim] ->(average) [target_sen * batch]
        for i in range(average_attention_gate.shape[0]):
            attention_gate_file.write(str(average_attention_gate[i][0]) + ' ')
        attention_gate_file.write('\n')
    attention_gate_file.close()


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          n_words_src=30000,
          n_words=30000,
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=1.,  # learning rate
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=80,
          saveto='model.npz',
          saveFreq=1000,  # save the parameters after every saveFreq updates
          validFreq=2500,
          dev_bleu_freq=20000,
          datasets=('/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
                    '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'),
          valid_datasets=('./data/dev/dev_en.tok',
                          './data/dev/dev_fr.tok'),
          small_train_datasets=('./data/train/small_en-fr.en',
                                './data/train/small_en-fr.fr'),
          test_datasets=('./data/test/test_en-fr.en.tok',
                         './data/test/test_en-fr.fr.tok'),
          use_dropout=False,
          reload_=False,
          overwrite=False,
          preload='',

          # Options below are from v-yanfa
          dump_before_train=True,
          plot_graph=None,
          vocab_filenames=('./data/dic/filtered_dic_en-fr.en.pkl',
                           './data/dic/filtered_dic_en-fr.fr.pkl'),
          map_filename='./data/dic/mapFullVocab2Top1MVocab.pkl',
          lr_discount_freq=80000,

          # Options of deeper encoder and decoder
          n_encoder_layers=1,
          n_decoder_layers=1,
          encoder_many_bidirectional=True,

          attention_layer_id=0,
          unit='gru',
          residual_enc=None,
          residual_dec=None,
          use_zigzag=False,

          initializer='orthogonal',
          given_embedding=None,

          dist_type=None,
          dist_recover_lr_iter=False,

          unit_size=2,
          cond_unit_size=2,
          gated_att=False,

          freeze_word_emb=False,
          only_word_att=False,
          visual_att=False,
          visual_att_gate=False,
          ft_by_cost=False,

          given_imm=False,
          dump_imm=False,
          shuffle_data=False,

          decoder_all_attention=False,
          average_context=False,
          task='en-fr',

          fine_tune_patience=8,
          nccl=False,
          src_vocab_map_file = None,
          tgt_vocab_map_file = None,

          trg_attention_layer_id=None,
          fix_dp_bug=False,
          io_buffer_size=40,
          ):
    model_options = locals().copy()

    # Set distributed computing environment
    worker_id = 0
    if dist_type == 'mv':
        try:
            import multiverso as mv
        except ImportError:
            from . import multiverso_ as mv

        worker_id = mv.worker_id()
    elif dist_type == 'mpi_reduce':
        from mpi4py import MPI
        mpi_communicator = MPI.COMM_WORLD
        worker_id = mpi_communicator.Get_rank()
        workers_cnt = mpi_communicator.Get_size()

        if nccl:
            nccl_comm = init_nccl_env(mpi_communicator)

    print 'Use {}, worker id: {}'.format('multiverso' if dist_type == 'mv' else 'mpi' if dist_recover_lr_iter else 'none', worker_id)
    sys.stdout.flush()

    # Set logging file
    set_logging_file('log/complete/e{}d{}_res{}_att{}_worker{}_task{}_{}.txt'.format(
        n_encoder_layers, n_decoder_layers, residual_enc, attention_layer_id,
        worker_id, task, time.strftime('%m-%d-%H-%M-%S'),
    ))

    log('''\
Start Time = {}
'''.format(
        time.strftime('%c'),
    ))

    # Model options: load and save
    message('Top options:')
    pprint(model_options)
    pprint(model_options, stream=get_logging_file())
    message('Done')
    sys.stdout.flush()

    load_options(model_options, reload_, preload, src_vocab_map_file and tgt_vocab_map_file)
    check_options(model_options)
    model_options['cost_normalization'] = 1
    ada_alpha = 0.95
    if dist_type == 'mpi_reduce':
        model_options['cost_normalization'] = workers_cnt

    message('Model options:')
    pprint(model_options)
    pprint(model_options, stream=get_logging_file())
    message()

    print 'Loading data'
    log('\n\n\nStart to prepare data\n@Current Time = {}'.format(time.time()))
    sys.stdout.flush()

    dataset_src, dataset_tgt = datasets[0], datasets[1]

    if shuffle_data:
        text_iterator_list = [None for _ in range(10)]
        text_iterator = None
    else:
        text_iterator_list = None
        text_iterator = TextIterator(
            dataset_src, dataset_tgt,
            vocab_filenames[0], vocab_filenames[1],
            batch_size,n_words_src, n_words,maxlen,k=io_buffer_size,
        )

    valid_iterator = TextIterator(
        valid_datasets[0], valid_datasets[1],
        vocab_filenames[0], vocab_filenames[1],
        valid_batch_size, n_words_src, n_words,k=io_buffer_size,
    )

    small_train_iterator = TextIterator(
        small_train_datasets[0], small_train_datasets[1],
        vocab_filenames[0], vocab_filenames[1],
        valid_batch_size, n_words_src, n_words,k=io_buffer_size,
    )

    test_iterator = SequentialTextIterator(
        test_datasets[0], test_datasets[1],
        vocab_filenames[0], vocab_filenames[1],
        1, n_words_src, n_words,
    )

    print 'Building model'
    if trg_attention_layer_id is None:
        model = NMTModel(model_options)
    else:
        model = TrgAttnNMTModel(model_options)

    params = model.initializer.init_params()

    # Reload parameters
    if reload_ and os.path.exists(preload):
        print 'Reloading model parameters'
        load_params(preload, params, src_map_file=src_vocab_map_file, tgt_map_file = tgt_vocab_map_file)
    sys.stdout.flush()

    # Given embedding
    if given_embedding is not None:
        print 'Loading given embedding...',
        load_embedding(params, given_embedding)
        print 'Done'

    print_params(params)

    model.init_tparams(params)

    # Build model
    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost, test_cost, x_emb, alpha, beta, attention_gate = model.build_model()   # return attention and gate.
    inps = [x, x_mask, y, y_mask]

    print 'Building sampler'
    f_init, f_next = model.build_sampler(trng=trng, use_noise=use_noise, batch_mode=True)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'
    sys.stdout.flush()
    test_cost = test_cost.mean()  # FIXME: do not regularize test_cost here

    cost = cost.mean()

    cost = l2_regularization(cost, model.P, decay_c)

    cost = regularize_alpha_weights(cost, alpha_c, model_options, x_mask, y_mask, opt_ret)

    print 'Building f_cost...',
    f_cost = theano.function(inps, test_cost, profile=profile)
    print 'Done'

    print 'Building f_att...',
    f_att = theano.function(inps, [alpha, beta], profile=profile)
    print 'Done'

    if gated_att:  # only for gated attention
        print 'Building f_att_gate...',
        average_attention_gate = tensor.mean(attention_gate, axis=-1)  # average the attention along the dimension
        f_att_gate = theano.function(inps, average_attention_gate, profile=profile)
        print 'Done'

    if plot_graph is not None:
        print 'Plotting post-compile graph...',
        theano.printing.pydotprint(
            f_cost,
            outfile='pictures/post_compile_{}'.format(plot_graph),
            var_with_name_simple=True,
        )
        print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(model.P, freeze_word_emb, only_word_att, gated_att))

    clip_shared = theano.shared(np.array(clip_c, dtype=fX), name='clip_shared')

    if dist_type != 'mpi_reduce':  # build grads clip into computational graph
        grads, g2 = clip_grad_remove_nan(grads, clip_shared, model.P, freeze_word_emb, only_word_att, gated_att)
    else:  # do the grads clip after gradients aggregation
        g2 = None

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',

    given_imm_data = get_adadelta_imm_data(optimizer, given_imm, preload)

    f_grad_shared, f_update, grads_shared, imm_shared = Optimizers[optimizer](
        lr, model.P, grads, inps, cost, g2=g2, given_imm_data=given_imm_data, alpha=ada_alpha,
        freeze_word_emb=freeze_word_emb, only_word_att=only_word_att, gated_att=gated_att)
    print 'Done'

    if dist_type == 'mpi_reduce':
        f_grads_clip = make_grads_clip_func(grads_shared=grads_shared, mt_tparams=model.P, clip_c_shared=clip_shared,
                                            freeze_word_emb=freeze_word_emb, only_word_att=only_word_att, gated_att=gated_att)

    print 'Optimization'
    log('Preparation Done\n@Current Time = {}'.format(time.time()))

    if dist_type == 'mv':
        mv.barrier()
    elif dist_type == 'mpi_reduce':
        # create receive buffers for mpi allreduce
        rec_grads = [np.zeros_like(p.get_value()) for p in model.P.itervalues()]

    estop = False
    history_errs = []
    best_bleu = -1.0
    best_valid_cost = 1e6
    best_p = None
    bad_counter = 0
    uidx = search_start_uidx(reload_, preload)

    epoch_n_batches = 0
    start_epoch = 0
    pass_batches = 0

    print 'worker', worker_id, 'uidx', uidx, 'l_rate', lrate, 'ada_alpha', ada_alpha, 'n_batches', epoch_n_batches, 'start_epoch', start_epoch, 'pass_batches', pass_batches

    start_uidx = uidx

    if dump_before_train:
        print 'Dumping before train...',
        saveto_uidx = '{}.iter{}.npz'.format(
            os.path.splitext(saveto)[0], uidx)
        np.savez(saveto_uidx, history_errs=history_errs,
                 uidx=uidx, **unzip(model.P))
        save_options(model_options, uidx, saveto)
        print 'Done'
        sys.stdout.flush()

    best_valid_cost = validation(valid_iterator, f_cost)
    small_train_cost = validation(small_train_iterator, f_cost)
    # best_bleu = translate_dev_get_bleu(model, f_init, f_next, trng, use_noise)
    if worker_id == 0:
        message('Initial Valid cost {:.5f} Small train cost {:.5f} Valid BLEU {:.2f}'.format(best_valid_cost, small_train_cost, best_bleu))

    commu_time_sum = 0.0
    cp_time_sum =0.0
    reduce_time_sum = 0.0

    start_time = time.time()
    finetune_cnt = 0

    if visual_att:
        visual_test_attention(test_iterator, f_att, f_att_gate)
        return
    if visual_att_gate:
        visual_test_attention_gate(test_iterator, f_att_gate)
        return

    for eidx in xrange(start_epoch, max_epochs):
        if shuffle_data:
            text_iterator = load_shuffle_text_iterator(
                eidx, worker_id, text_iterator_list,
                datasets, vocab_filenames, batch_size, maxlen, n_words_src, n_words, buffer_size=io_buffer_size,
            )
        n_samples = 0
        if dist_type == 'mpi_reduce':
            mpi_communicator.Barrier()

        for i, (x, y) in enumerate(text_iterator):
            if eidx == start_epoch and i < pass_batches:  # ignore the first several batches when reload
                continue
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            effective_uidx = uidx - start_uidx
            ud_start = time.time()

            # compute cost, grads
            if dist_type != 'mpi_reduce':
                cost, g2_value = f_grad_shared(x, x_mask, y, y_mask)
            else:
                cost = f_grad_shared(x, x_mask, y, y_mask)

            if dist_type == 'mpi_reduce':
                reduce_start = time.time()
                commu_time = 0
                gpucpu_cp_time = 0
                if not nccl:
                    commu_time, gpucpu_cp_time = all_reduce_params(grads_shared, rec_grads)
                else:
                    commu_time, gpucpu_cp_time = all_reduce_params_nccl(nccl_comm, grads_shared)
                reduce_time = time.time() - reduce_start
                commu_time_sum += commu_time
                reduce_time_sum += reduce_time
                cp_time_sum += gpucpu_cp_time

                g2_value = f_grads_clip()
                print '@Worker = {}, Reduce time = {:.5f}, Commu time = {:.5f}, Copy time = {:.5f}'.format(worker_id, reduce_time, commu_time, gpucpu_cp_time)

            curr_lr = lrate if not dist_type or dist_recover_lr_iter < effective_uidx else lrate * 0.05 + effective_uidx * lrate / dist_recover_lr_iter * 0.95
            if curr_lr < lrate:
                print 'Curr lr {:.3f}'.format(curr_lr)

            # do the update on parameters
            f_update(curr_lr)

            ud = time.time() - ud_start

            if np.isnan(cost) or np.isinf(cost):
                message('NaN detected')
                sys.stdout.flush()
                clip_shared.set_value(clip_shared.get_value() * 0.9)
                message('Discount clip value to {:.4f} at iteration {}'.format(clip_shared.get_value(), uidx))

                # reload the best saved model
                if not os.path.exists(saveto):
                    message('No saved model at {}. Task exited'.format(saveto))
                    return 1., 1., 1.
                else:
                    message('Load previously dumped model at {}'.format(saveto))
                    prev_params = load_params(saveto, params)
                    zipp(prev_params, model.P)
                    prev_imm_data = get_adadelta_imm_data(optimizer, True, saveto)
                    adadelta_set_imm_data(optimizer, prev_imm_data, imm_shared)

            # discount learning rate
            # FIXME: Do NOT enable this and fine-tune at the same time
            if lr_discount_freq > 0 and np.mod(effective_uidx, lr_discount_freq) == 0:
                lrate *= 0.5
                message('Discount learning rate to {} at iteration {}'.format(lrate, uidx))

            # sync batch
            if dist_type == 'mv' and np.mod(uidx, dispFreq) == 0:
                comm_start = time.time()
                model.sync_tparams()
                message('@Comm time = {:.5f}'.format(time.time() - comm_start))

            # verbose
            if np.mod(uidx, dispFreq) == 0:
                message('Worker {} Epoch {} Update {} Cost {:.5f} G2 {:.5f} UD {:.5f} Time {:.5f} s'.format(
                    worker_id, eidx, uidx, float(cost), float(g2_value), ud, time.time() - start_time,
                ))
                sys.stdout.flush()

            if np.mod(uidx, saveFreq) == 0 and worker_id == 0:
                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    model.save_model(saveto, history_errs, uidx)
                    print 'Done'
                    sys.stdout.flush()

                # save immediate data in adadelta
                saveto_imm_path = '{}_latest.npz'.format(os.path.splitext(saveto)[0])
                dump_adadelta_imm_data(optimizer, imm_shared, dump_imm, saveto_imm_path)

            if np.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_cost = validation(valid_iterator, f_cost)
                small_train_cost = validation(small_train_iterator, f_cost)
                valid_bleu = translate_dev_get_bleu(model, f_init, f_next, trng, use_noise)
                if worker_id == 0:
                    message('Valid cost {:.5f} Small train cost {:.5f} Valid BLEU {:.2f}'.format(valid_cost,
                                                                                                 small_train_cost,
                                                                                                 valid_bleu))
                use_noise.set_value(1.)
                sys.stdout.flush()

                # Fine-tune based on dev bleu
                if fine_tune_patience > 0:
                    if ft_by_cost:
                        if valid_cost > best_valid_cost:
                            bad_counter = 0
                            best_valid_cost = valid_cost
                            if worker_id == 0:
                                message('Dump the the best model so far at uidx {}'.format(uidx))
                                model.save_model(saveto, history_errs)
                                dump_adadelta_imm_data(optimizer, imm_shared, dump_imm, saveto)
                        else:
                            bad_counter += 1
                            if bad_counter >= fine_tune_patience:
                                print 'Fine tune:',
                                if finetune_cnt % 2 == 0:
                                    lrate = np.float32(lrate * 0.2)
                                    message('Discount learning rate to {} at iteration {}'.format(lrate, uidx))
                                else:
                                    clip_shared.set_value(np.float32(clip_shared.get_value() * 0.25))
                                    message('Discount clip value to {} at iteration {}'.format(clip_shared.get_value(),
                                                                                               uidx))
                                finetune_cnt += 1
                                bad_counter = 0
                    else:
                        if valid_bleu > best_bleu:
                            bad_counter = 0
                            best_bleu = valid_bleu
                            # dump the best model so far, including the immediate file
                            if worker_id == 0:
                                message('Dump the the best model so far at uidx {}'.format(uidx))
                                model.save_model(saveto, history_errs)
                                dump_adadelta_imm_data(optimizer, imm_shared, dump_imm, saveto)
                        else:
                            bad_counter += 1
                            if bad_counter >= fine_tune_patience:
                                print 'Fine tune:',
                                if finetune_cnt % 2 == 0:
                                    lrate = np.float32(lrate * 0.5)
                                    message('Discount learning rate to {} at iteration {}'.format(lrate, uidx))
                                    # if lrate <= 0.025:
                                    #     message('Learning rate decayed to {:.5f}, task completed'.format(lrate))
                                    #     return 1., 1., 1.
                                else:
                                    clip_shared.set_value(np.float32(clip_shared.get_value() * 0.25))
                                    message('Discount clip value to {} at iteration {}'.format(clip_shared.get_value(), uidx))
                                finetune_cnt += 1
                                bad_counter = 0

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after {} iterations!'.format(uidx)
                estop = True
                break

        print 'Seen {} samples'.format(n_samples)

        if estop:
            break

    if best_p is not None:
        zipp(best_p, model.P)

    use_noise.set_value(0.)

    return 0.


if __name__ == '__main__':
    pass
