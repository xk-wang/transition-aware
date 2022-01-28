from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import os
from multiprocessing import Manager, Pool  #因为队列被多个进程共享，必须使用Manager里面的Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from glovar import get_annotations
from glovar import cqt_dual, params
from glovar import MODES, FLAGS
from glovar import _int_feature, _float_feature

from functools import partial
import time
import random
random.seed(1)


def open_tfrecord_writer(network, mode, parallel_num):
    tfrd_dir = os.path.join(FLAGS.save_dir, mode, network)
    if not os.path.exists(tfrd_dir):
        os.makedirs(tfrd_dir)

    writers=[tf.python_io.TFRecordWriter(os.path.join(tfrd_dir, '%04d.tfrecords'%i)) for i in range(parallel_num)]

    return writers


def close_tfrecord_writer(writers):
    for writer in writers:
        writer.close()

def frame_labels(label_path):

    sr = params.sr
    hop_len = params.hop_len

    annotation = np.loadtxt(label_path)[:, [0, 2]]
    frame_len = int(annotation[-1, 0]*sr/hop_len)
    pitch_labels = np.zeros([frame_len, 88], dtype=np.uint8)
    annotation = annotation[annotation[:,0]<=frame_len*hop_len/sr]

    for onset, pitch in annotation:
        pitch_labels[int(onset*sr/hop_len), int(pitch-21)] = 1

    onset_labels = pitch_labels.any(axis=-1, keepdims=True).astype(np.uint8)
    weighted_onset_labels = np.zeros_like(onset_labels)

    frame_len = onset_labels.shape[0]
    for i in range(frame_len):
        if onset_labels[i] == 1:
            weighted_onset_labels[i] = 3
        elif onset_labels[max(i-1, 0):min(i+2, frame_len)].any():
            weighted_onset_labels[i] = 2
        elif onset_labels[max(i-2, 0):min(i+3, frame_len)].any():
            weighted_onset_labels[i] = 1

    return weighted_onset_labels, pitch_labels


def get_train_examples(mode, specs, onset_labels, pitch_labels):

    spec_len, num_data, depth = specs.shape
    num_data = min(num_data, onset_labels.shape[0])
    offset = params.win_len//2
    specs = np.pad(specs, ((0, 0), (offset, offset), (0,0)), 'constant')
    onset_labels = np.pad(onset_labels, ((offset, offset), (0, 0)), 'constant')
    pitch_labels = np.pad(pitch_labels, ((offset, offset), (0, 0)), 'constant')

    split_specs = np.zeros([num_data, spec_len, params.win_len, depth], dtype=np.float32)
    split_onset_labels = np.zeros([num_data, 1], dtype=np.uint8)
    split_pitch_labels = np.zeros([num_data, 88], dtype=np.uint8)

    for i in range(offset, offset+num_data):
        split_specs[i-offset] = specs[:, i-offset:i+offset+1]
        split_onset_labels[i-offset] = onset_labels[i]
        split_pitch_labels[i-offset] = pitch_labels[i]

    pos_idxs = list(np.where(np.reshape(split_onset_labels, [-1]) == 3)[0])
    neg_idxs = list(np.where(np.reshape(split_onset_labels, [-1]) < 3)[0])

    sample_neg_idxs = random.sample(neg_idxs,len(neg_idxs)//10) if mode == tf.estimator.ModeKeys.TRAIN else neg_idxs
    onset_idxs = pos_idxs + sample_neg_idxs
    pitch_idxs = pos_idxs
    random.shuffle(onset_idxs)
    random.shuffle(pitch_idxs)

    onset_specs, onset_labels = split_specs[onset_idxs], split_onset_labels[onset_idxs]
    pitch_specs, pitch_labels = split_specs[pitch_idxs], split_pitch_labels[pitch_idxs]

    return (onset_specs, onset_labels), (pitch_specs, pitch_labels)


def producer(q1, q2, mode, annotation):
    print('\n', annotation[0])
    wav_path, label_path = annotation
    specs = cqt_dual(wav_path)
    onset_labels, pitch_labels = frame_labels(label_path)

    (onset_specs, onset_labels), (pitch_specs, pitch_labels) = get_train_examples(mode, specs, onset_labels, pitch_labels)

    to_example = lambda spec, label: tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'spec': _float_feature(spec),
                            'label': _int_feature(label),
                            }
                        )
                    )

    def writer_to_queue(specs, labels, q):
        for spec, label in zip(specs, labels):
            example = to_example(spec, label).SerializeToString()
            q.put(example)

    t1 = Thread(target = writer_to_queue, args = (onset_specs, onset_labels, q1))
    t2 = Thread(target = writer_to_queue, args = (pitch_specs, pitch_labels, q2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print('\n', 'processing', os.getpid(), 'is leaving')


def consumer(q1, q2, w):
    def writer_tfrd(q, w):
        time.sleep(240)
        while True:
            try:
                example = q.get(timeout=300)
            except Exception as e:
                break
            w.write(example)
            time.sleep(0.01)
                        
    w1, w2 = w
    t1 = Thread(target=writer_tfrd, args=(q1, w1))
    t2 = Thread(target=writer_tfrd, args=(q2, w2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print('\n', 'consumer', os.getpid(), 'is leaving')


def convert_to_tfrecord(mode, anno):

    assert mode in MODES, "模式错误"

    onset_writers = open_tfrecord_writer('onset', mode, 64)
    pitch_writers = open_tfrecord_writer('pitch', mode, 64)

    writers = list(zip(onset_writers, pitch_writers))

    manager = Manager()
    q1 = manager.Queue()
    q2 = manager.Queue()

    p1 = Pool(FLAGS.parallel_num)
    p2 = ThreadPoolExecutor(64)

    p1_func = partial(producer, q1, q2, mode)
    p2_func = partial(consumer, q1, q2)

    p1.map_async(p1_func, anno)
    p2.map(p2_func, writers)
    p1.close()
    p1.join()
    print('\nfinish process pool')
    p2.shutdown()
    print('\nfinish threading pool')

    close_tfrecord_writer(onset_writers)
    close_tfrecord_writer(pitch_writers)

    print('\nreturn to main thread')


def main(_):
    
    annotation = get_annotations(tf.estimator.ModeKeys.TRAIN)
    convert_to_tfrecord(tf.estimator.ModeKeys.TRAIN, annotation)

    annotation = get_annotations(tf.estimator.ModeKeys.EVAL)
    convert_to_tfrecord(tf.estimator.ModeKeys.EVAL, annotation)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()