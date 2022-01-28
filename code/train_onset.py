from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import math

from models import onset_model, onset_loss, onset_metric
from glovar import FLAGS, input_fn
FLAGS.network = 'onset'

train_examples = 7964564
save_checkpoints_steps = 2000
initial_lr = 0.01
train_dir = '/dataset/new_maestro-v1_tfrd/train/onset'
# evaluate_dir = '/dataset/new_maestro-v1_tfrd/eval/onset'
evaluate_dir = '/dataset/new_maestro-v1_tfrd/eval/onset_mini'

  
def model_fn(features, labels, mode):

    logits = onset_model(features, mode)

    predictions = {
        'probabilities': tf.nn.sigmoid(logits, name='sigmoid_tensor'),  
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = onset_loss(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate=tf.train.exponential_decay(initial_lr, global_step, decay_steps=500, decay_rate=0.98, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tf.summary.scalar('learning_rate', learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    metrics = onset_metric(labels, predictions['probabilities'])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def main(_):

    logging_hook1 = tf.train.LoggingTensorHook(
        every_n_iter=500,
        tensors={
            'onset_f': 'f1_score/update_op',
            'onset_p': 'precision/update_op',
            'onset_r': 'recall/update_op',
            'onset_auc': 'auc/update_op',
            'loss': 'weighted_loss/loss'
        }
    )
    logging_hook2 = tf.train.LoggingTensorHook(
        every_n_iter=500,
        tensors={
            'onset_f': 'f1_score/update_op',
            'onset_p': 'precision/update_op',
            'onset_r': 'recall/update_op',
            'onset_auc': 'auc/update_op',
            'loss': 'weighted_loss/loss'
        }
    )

    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    run_config = tf.estimator.RunConfig(session_config=sess_config, tf_random_seed=1, save_checkpoints_steps=save_checkpoints_steps, 
                                        keep_checkpoint_max=None, save_summary_steps=100, log_step_count_steps=500)

    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, config=run_config)

    # train_dir_num = len(train_dir)
    # train_dir_no = 0 #9之后都有问题 1，2，3有问题
    # for i in range(FLAGS.epochs*train_dir_num):
    #     print('\n', '='*10,train_dir[train_dir_no],'='*10, '\n')
        
    #     model.train(input_fn=lambda: input_fn(train_dir[train_dir_no], tf.estimator.ModeKeys.TRAIN, FLAGS.batch_size),
    #                 steps=steps_per_dir,
    #                 hooks=[logging_hook1])
    #     print('\n', '=' * 10, 'Evaluating after training for %d steps'%steps_per_dir, '=' * 10)
    #     model.evaluate(input_fn=lambda: input_fn(evaluate_dir, tf.estimator.ModeKeys.EVAL, 1024),
    #                    steps=None,
    #                    hooks=[logging_hook2])
    #     train_dir_no = (train_dir_no+1)%train_dir_num

    max_steps = int(math.ceil(FLAGS.epochs*train_examples/FLAGS.batch_size/save_checkpoints_steps)*save_checkpoints_steps)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_dir, tf.estimator.ModeKeys.TRAIN), 
                                        max_steps=max_steps, 
                                        hooks=[logging_hook1])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(evaluate_dir, tf.estimator.ModeKeys.EVAL),
                                      steps=None,
                                      hooks=[logging_hook2],
                                      throttle_secs=10)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()