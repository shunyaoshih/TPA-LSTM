import logging
import numpy as np
from tqdm import tqdm

def test(para, sess, model, data_generator):
    sess.run(data_generator.iterator.initializer)

    test_rse = 0.0
    count = 0
    n_samples = 0
    all_outputs, all_labels = [], []

    tp, fp, tn, fn = 0, 0, 0, 0
    while True:
        try:
            [outputs, labels] = sess.run(
                fetches=[
                    model.all_rnn_outputs,
                    model.labels
                ]
            )
            if para.mts:
                test_rse += np.sum(
                    ((outputs - labels) * data_generator.scale) ** 2
                )
                all_outputs.append(outputs)
                all_labels.append(labels)
            elif para.data_set == 'muse' or para.data_set == 'lpd5':
                for b in range(para.batch_size):
                    for p in range(128):
                        if outputs[b][p] >= 0.5 and labels[b][p] >= 0.5:
                            tp += 1
                        elif outputs[b][p] >= 0.5 and labels[b][p] < 0.5:
                            fp += 1
                        elif outputs[b][p] < 0.5 and labels[b][p] < 0.5:
                            tn += 1
                        elif outputs[b][p] < 0.5 and labels[b][p] >= 0.5:
                            fn += 1
            count += 1
            n_samples += np.prod(outputs.shape)
        except:
            break
    if para.mts:
        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)
        sigma_outputs = all_outputs.std(axis=0)
        sigma_labels = all_labels.std(axis=0)
        mean_outputs = all_outputs.mean(axis=0)
        mean_labels = all_labels.mean(axis=0)
        idx = sigma_labels != 0
        test_corr = (
            (all_outputs - mean_outputs) * (all_labels - mean_labels)
        ).mean(axis=0) / (sigma_outputs * sigma_labels)
        test_corr = test_corr[idx].mean()
        test_rse = (
            np.sqrt(test_rse / n_samples) / data_generator.rse
        )
        logging.info("test rse: %.5f, test corr: %.5f" % (test_rse, test_corr))
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall >= 1e-6:
            F1 = 2 * precision * recall / (precision + recall)
        else:
            F1 = 0.0
        logging.info('# of testing data: %d' % count * para.batch_size)
        logging.info('precision: %.5f' % precision)
        logging.info('recall: %.5f' % recall)
        logging.info('F1 score: %.5f' % F1)
