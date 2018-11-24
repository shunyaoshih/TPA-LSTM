import os
import tensorflow as tf

from lib.setup import params_setup, logging_config_setup, config_setup
from lib.model_utils import create_graph, load_weights, print_num_of_trainable_parameters
from lib.train import train
from lib.test import test


def main():
    para = params_setup()
    logging_config_setup(para)

    graph, model, data_generator = create_graph(para)

    with tf.Session(config=config_setup(), graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        load_weights(para, sess, model)
        print_num_of_trainable_parameters()

        try:
            if para.mode == 'train':
                train(para, sess, model, data_generator)
            elif para.mode == 'test':
                test(para, sess, model, data_generator)

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            print('Stop')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
