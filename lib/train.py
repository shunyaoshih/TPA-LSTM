import time
import logging
import numpy as np
import tensorflow as tf

from lib.setup import config_setup
from lib.model_utils import save_model, create_valid_graph, load_weights


def train(para, sess, model, train_data_generator):
    valid_para, valid_graph, valid_model, valid_data_generator = \
        create_valid_graph(para)

    with tf.Session(config=config_setup(), graph=valid_graph) as valid_sess:
        valid_sess.run(tf.global_variables_initializer())

        for epoch in range(1, para.num_epochs + 1):
            logging.info("Epoch: %d" % epoch)
            sess.run(train_data_generator.iterator.initializer)

            start_time = time.time()
            train_loss = 0.0
            count = 0
            while True:
                try:
                    [loss, global_step, _] = sess.run(
                        fetches=[model.loss, model.global_step, model.update])
                    train_loss += loss
                    count += 1
                except tf.errors.OutOfRangeError:
                    logging.info(
                        "global step: %d, loss: %.5f, epoch time: %.3f",
                        global_step, train_loss / count,
                        time.time() - start_time)
                    save_model(para, sess, model)
                    break

            # validation
            load_weights(valid_para, valid_sess, valid_model)
            valid_sess.run(valid_data_generator.iterator.initializer)
            valid_loss = 0.0
            valid_rse = 0.0
            count = 0
            n_samples = 0
            all_outputs, all_labels = [], []
            while True:
                try:
                    [loss, outputs, labels] = valid_sess.run(fetches=[
                        valid_model.loss,
                        valid_model.all_rnn_outputs,
                        valid_model.labels,
                    ])
                    if para.mts:
                        valid_rse += np.sum(
                            ((outputs - labels) * valid_data_generator.scale)
                            **2)
                        all_outputs.append(outputs)
                        all_labels.append(labels)
                        n_samples += np.prod(outputs.shape)
                    valid_loss += loss
                    count += 1
                except tf.errors.OutOfRangeError:
                    break
            if para.mts:
                all_outputs = np.concatenate(all_outputs)
                all_labels = np.concatenate(all_labels)
                sigma_outputs = all_outputs.std(axis=0)
                sigma_labels = all_labels.std(axis=0)
                mean_outputs = all_outputs.mean(axis=0)
                mean_labels = all_labels.mean(axis=0)
                idx = sigma_labels != 0
                valid_corr = ((all_outputs - mean_outputs) *
                              (all_labels - mean_labels)).mean(
                                  axis=0) / (sigma_outputs * sigma_labels)
                valid_corr = valid_corr[idx].mean()
                valid_rse = (
                    np.sqrt(valid_rse / n_samples) / train_data_generator.rse)
                valid_loss /= count
                logging.info(
                    "validation loss: %.5f, validation rse: %.5f, validation corr: %.5f",
                    valid_loss, valid_rse, valid_corr)
            else:
                logging.info("validation loss: %.5f", valid_loss / count)
