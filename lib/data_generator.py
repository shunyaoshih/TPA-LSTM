from abc import ABCMeta, abstractmethod
import glob
import logging
import os
import math
from shutil import copyfile
import tarfile
from tqdm import tqdm

import numpy as np
import pypianoroll as ppr
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin

from lib.utils import check_path_exists, create_dir
from lib.utils import download_file_from_google_drive, download_file


class DataGenerator(metaclass=ABCMeta):
    def __init__(self, para):
        self.DIRECTORY = "./data"
        self.para = para
        self.iterator = None

    def inputs(self, mode, batch_size, num_epochs=None):
        """Reads input data num_epochs times.
        Args:
        mode: String for the corresponding tfrecords ('train', 'validation', 'test')
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
        train forever.
        Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, 28, 28]
        in the range [0.0, 1.0].
        * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
        """
        if mode != "train" and mode != "validation" and mode != "test":
            raise ValueError("mode: {} while mode should be "
                             "'train', 'validation', or 'test'".format(mode))

        filename = self.DATA_PATH + "/" + mode + ".tfrecords"
        logging.info("Loading data from {}".format(filename))

        with tf.name_scope("input"):
            # TFRecordDataset opens a binary file and
            # reads one record at a time.
            # `filename` could also be a list of filenames,
            # which will be read in order.
            dataset = tf.data.TFRecordDataset(filename)

            # The map transformation takes a function and
            # applies it to every element
            # of the dataset.
            dataset = dataset.map(self._decode)
            for f in self._get_map_functions():
                dataset = dataset.map(f)

            # The shuffle transformation uses a finite-sized buffer to shuffle
            # elements in memory. The parameter is the number of elements in the
            # buffer. For completely uniform shuffling, set the parameter to be
            # the same as the number of elements in the dataset.
            if self.para.mode == "train":
                dataset = dataset.shuffle(1000 + 3 * batch_size)

            # dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(batch_size)

            self.iterator = dataset.make_initializable_iterator()
            return self.iterator.get_next()

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @abstractmethod
    def _decode(self, serialized_example):
        pass

    @abstractmethod
    def _get_map_functions(self):
        pass


class MusicDataGenerator(DataGenerator):
    def __init__(self, para):
        DataGenerator.__init__(self, para)

        self.DATA_PATH = None
        self.FILENAME = None
        self.DATA_FULL_PATH = None
        self.DATASET_ID = None

        para.input_size = self.INPUT_SIZE = 128
        para.max_len = self.MAX_LEN = 16
        para.output_size = self.OUTPUT_SIZE = 128
        para.total_len = 64

    def _download_file(self):
        logging.info(
            "Downloading %s dataset from Google drive..." % self.para.data_set)
        create_dir(self.DATA_PATH)
        download_file_from_google_drive(self.DATASET_ID,
                                        self.DATA_FULL_PATH + ".tar")

    def _extract_file(self):
        if not check_path_exists(self.DATA_FULL_PATH):
            logging.info("Extracting %s dataset..." % self.para.data_set)
            tarfile.open(self.DATA_FULL_PATH + ".tar").extractall(
                path=self.DATA_PATH)

    def _decode(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={"pianoroll": tf.VarLenFeature(tf.int64)},
        )
        pianoroll = tf.sparse_tensor_to_dense(features["pianoroll"])

        pianoroll = tf.cast(
            tf.reshape(pianoroll, [self.MAX_LEN + 1, self.INPUT_SIZE]),
            tf.float32,
        )
        return pianoroll

    def _get_map_functions(self):
        return [self._augment]

    def _augment(self, pianoroll):
        # pianoroll: [self.MAX_LEN + 1, self.OUTPUT_SIZE]
        rnn_input = tf.slice(pianoroll, [0, 0],
                             [self.MAX_LEN, self.OUTPUT_SIZE])
        rnn_input_len = tf.constant(self.MAX_LEN, dtype=tf.int32)
        target_output = tf.slice(pianoroll, [1, 0],
                                 [self.MAX_LEN, self.OUTPUT_SIZE])
        return rnn_input, rnn_input_len, target_output

    def _convert_to_tfrecords(self, mode, filename_list):
        filename = self.DATA_PATH + "/" + mode + ".tfrecords"
        if check_path_exists(filename):
            return

        logging.info("Writing {}".format(filename))
        with tf.python_io.TFRecordWriter(filename) as writer:
            for filename in tqdm(filename_list):
                if filename.endswith(".mid") or filename.endswith(".midi"):
                    multi_track = ppr.parse(filename)
                else:
                    multi_track = ppr.load(filename)

                TOTAL_STEPS = self._choose_total_steps(multi_track)
                if TOTAL_STEPS == 1e8:
                    continue
                RANGE = self.INPUT_SIZE
                FINAL_STEPS = math.ceil(TOTAL_STEPS / 24)
                multi_data = np.zeros((FINAL_STEPS, RANGE))

                for track in multi_track.tracks:
                    if not self._is_valid_track(track):
                        continue
                    data = track.pianoroll.astype(int)
                    data = self._sampling(data)
                    multi_data = np.add(multi_data, data)
                multi_data = np.clip(multi_data, 0, 1).astype(int)

                RANGE = self._split_into_segments(multi_data, 1)
                length = self.MAX_LEN

                for start in RANGE:
                    end = start + length
                    if end >= FINAL_STEPS:
                        break
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "pianoroll":
                                self._int64_list_feature(
                                    multi_data[start:end + 1].flatten())
                            }))
                    writer.write(example.SerializeToString())

    def _choose_total_steps(self, multi_track):
        TOTAL_STEPS = 1e8
        for track in multi_track.tracks:
            now_steps = track.pianoroll.shape[0]
            if not self._is_valid_track(track):
                continue
            TOTAL_STEPS = min(TOTAL_STEPS, now_steps)
        return TOTAL_STEPS

    def _is_valid_track(self, track):
        if track.is_drum:
            return False
        if track.pianoroll.shape[0] == 0:
            return False
        return True

    def _sampling(self, data):
        return data[::24]

    def _split_into_segments(self, data, step):
        return range(0, data.shape[0], step)


class MIDIDataGenerator(MusicDataGenerator):
    def __init__(self, para):
        MusicDataGenerator.__init__(self, para)

    def _parse_to_tfrecords(self):
        for mode in ["train", "validation", "test"]:
            filename_list = []
            for file_type in ["mid", "midi"]:
                path = self.DATA_FULL_PATH + "/" + mode + "/*." + file_type
                filename_list += glob.glob(path, recursive=True)
            self._convert_to_tfrecords(mode, filename_list)


class LPD5DataGenerator(MusicDataGenerator):
    def __init__(self, para):
        MusicDataGenerator.__init__(self, para)

        self.DATA_PATH = self.DIRECTORY + "/lpd5_data"
        self.FILENAME = "lpd_5_cleansed"
        self.DATA_FULL_PATH = self.DATA_PATH + "/" + self.FILENAME
        self.DATASET_ID = "1IkfgCwJ6fCjGYLMelDrKV7mGSNF8llex"

        self._download_file()
        self._extract_file()
        self._parse_to_tfrecords()

    def _parse_to_tfrecords(self):
        filename_list = glob.glob(
            self.DATA_FULL_PATH + "/**/*.npz", recursive=True)
        num_of_raw_data = len(filename_list)

        l, r = int(num_of_raw_data * 0.8), int(num_of_raw_data * 0.9)
        self._convert_to_tfrecords("train", filename_list[:l])
        self._convert_to_tfrecords("validation", filename_list[l:r])
        self._convert_to_tfrecords("test", filename_list[r:])

    def _split_into_segments(self, data, step):
        """
        pianoroll: [num_of_time_steps, 128]
        segments: list of tuple(start, end)
        """
        TOTAL_STEPS, RANGE = data.shape
        segments = []

        num_of_notes = np.sum(data, axis=1)

        for start in range(0, TOTAL_STEPS, step):
            if start + self.MAX_LEN >= TOTAL_STEPS:
                break
            num_of_total_notes = np.sum(
                num_of_notes[start:start + self.MAX_LEN])
            if num_of_total_notes >= 2.0 * self.MAX_LEN:
                segments.append(start)
        return segments


class MuseDataGenerator(MIDIDataGenerator):
    def __init__(self, para):
        MIDIDataGenerator.__init__(self, para)

        self.DATA_PATH = self.DIRECTORY + "/muse_data"
        self.FILENAME = "MuseData"
        self.DATA_FULL_PATH = self.DATA_PATH + "/" + self.FILENAME
        self.DATASET_ID = "1a5361IfxxEY1mmTfqAviiIkq6u2OYFJ7"

        self._download_file()
        self._extract_file()
        self._parse_to_tfrecords()


class TimeSeriesDataGenerator(DataGenerator):
    def __init__(self, para):
        DataGenerator.__init__(self, para)
        self.h = para.horizon
        self.DATA_PATH = os.path.join(self.DIRECTORY,
                                      para.data_set + str(self.h))
        create_dir(self.DATA_PATH)
        self._download_file()
        self.split = [0, 0.6, 0.8, 1]
        self.split_names = ["train", "validation", "test"]
        self._preprocess(para)
        del self.raw_dat, self.dat

    def _download_file(self):
        logging.info("Downloading Time Series {} data set...".format(
            self.para.data_set))
        url = "https://github.com/laiguokun/multivariate-time-series-data/"
        url += "blob/master/{}/".format(self.para.data_set)
        if self.para.data_set == "solar-energy":
            url += "solar_AL.txt.gz?raw=true"
        else:
            url += "{}.txt.gz?raw=true".format(self.para.data_set)
        self.out_fn = os.path.join(self.DATA_PATH,
                                   self.para.data_set + ".txt.gz")
        download_file(url, self.out_fn)

    def _preprocess(self, para):

        self.raw_dat = np.loadtxt(self.out_fn, delimiter=",")
        para.input_size = self.INPUT_SIZE = self.raw_dat.shape[1]
        self.rse = self._compute_rse()

        para.max_len = self.MAX_LEN = self.para.highway
        assert self.para.highway == self.para.attention_len
        para.output_size = self.OUTPUT_SIZE = self.raw_dat.shape[1]
        para.total_len = self.TOTAL_LEN = 1
        self.dat = np.zeros(self.raw_dat.shape)
        self.scale = np.ones(self.INPUT_SIZE)
        for i in range(self.INPUT_SIZE):
            mn = np.min(self.raw_dat[:, i])
            if para.data_set == 'electricity':
                self.scale[i] = np.max(self.raw_dat[:, i]) - mn
            else:
                self.scale[i] = np.max(self.raw_dat) - mn
            self.dat[:, i] = (self.raw_dat[:, i] - mn) / self.scale[i]
        logging.info('rse = {}'.format(self.rse))
        for i in range(len(self.split) - 1):
            self._convert_to_tfrecords(self.split[i], self.split[i + 1],
                                       self.split_names[i])

    def _compute_rse(self):
        st = int(self.raw_dat.shape[0] * self.split[2])
        ed = int(self.raw_dat.shape[0] * self.split[3])
        Y = np.zeros((ed - st, self.INPUT_SIZE))
        for target in range(st, ed):
            Y[target - st] = self.raw_dat[target]
        return np.std(Y)

    def _convert_to_tfrecords(self, st, ed, name):
        st = int(self.dat.shape[0] * st)
        ed = int(self.dat.shape[0] * ed)
        out_fn = os.path.join(self.DATA_PATH, name + ".tfrecords")
        if check_path_exists(out_fn):
            return
        with tf.python_io.TFRecordWriter(out_fn) as record_writer:
            for target in tqdm(range(st, ed)):
                end = target - self.h + 1
                beg = end - self.para.max_len
                if beg < 0:
                    continue
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "x":
                            self._float_list_feature(self.dat[beg:end].
                                                     flatten()),
                            "y":
                            self._float_list_feature(self.dat[target]),
                        }))
                record_writer.write(example.SerializeToString())

    def _get_map_functions(self):
        return []

    def _decode(self, serialized_example):
        example = tf.parse_single_example(
            serialized_example,
            features={
                "x":
                tf.FixedLenFeature([self.MAX_LEN, self.INPUT_SIZE],
                                   tf.float32),
                "y":
                tf.FixedLenFeature([self.OUTPUT_SIZE], tf.float32),
            },
        )
        rnn_input = tf.to_float(
            tf.reshape(example["x"], (self.MAX_LEN, self.INPUT_SIZE)))
        rnn_input_len = tf.constant(self.MAX_LEN, dtype=tf.int32)
        target_output = tf.expand_dims(tf.to_float(example["y"]), 0)
        target_output = tf.tile(target_output, [self.MAX_LEN, 1])
        return rnn_input, rnn_input_len, target_output
