#-*----功能----*-#
#目前在谷歌数据集上由于预测集性能表现比较好，而训练集的精度不是标准的验证精度
#因此需要重新进行评估工作

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from glovar import input_fn, FLAGS
from models import onset_model, onset_loss, onset_metric
from models import pitch_model, pitch_loss, pitch_metric


def model_fn(features, labels, mode):

    if FLAGS.network == 'onset':
      model = onset_model
      loss_func = onset_loss
      metrics = onset_metric
    elif FLAGS.network == 'pitch':
      model = pitch_model
      loss_func = pitch_loss
      metrics = pitch_metric

    logits = model(features, mode)

    predictions = {
        'probabilities': tf.nn.sigmoid(logits, name='sigmoid_tensor'),  
    }

    loss = loss_func(labels, logits)
    metrics = metrics(labels, predictions['probabilities'])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=None,
        eval_metric_ops=metrics)


def main(_):

    logging_hook = tf.train.LoggingTensorHook(
        every_n_iter=500,
        tensors={
            'pitch_f': 'f1_score/update_op',
            'pitch_p': 'precision/update_op',
            'pitch_r': 'recall/update_op',
            'pitch_auc': 'auc/update_op',
            'loss': 'weighted_loss/loss'
        }
    )

    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    run_config = tf.estimator.RunConfig(session_config=sess_config, save_summary_steps=100, log_step_count_steps=500)

    checkpoint_paths = [x.replace('.meta', '') for x in os.listdir(FLAGS.model_dir) if x.endswith('.meta')]
    checkpoint_paths = sorted(checkpoint_paths, key=lambda x: int(x.split('-')[1]))
    checkpoint_paths = [os.path.join(FLAGS.model_dir, x) for x in checkpoint_paths][-1:]

    model = tf.estimator.Estimator(model_fn=model_fn, config=run_config, model_dir=FLAGS.save_dir)

    for checkpoint_path in checkpoint_paths:
      model.evaluate(input_fn=lambda: input_fn(FLAGS.tfrd_dir, tf.estimator.ModeKeys.EVAL), steps=None, hooks=[logging_hook],
                     checkpoint_path=checkpoint_path)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()