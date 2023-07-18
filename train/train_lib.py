"""Helper training functions."""

import functools
import tensorflow as tf
from tensorflow_addons import metrics as tfa_metrics
from tensorflow_addons import optimizers as tfa_optimizers


def get_linear_model(input_dim: int, output_dim: int):
  return tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(input_dim,)),
      tf.keras.layers.Dense(output_dim),
  ])


class LinearFineTuneModel(tf.keras.Model):
  def __init__(self, backbone, output_dim):
    super().__init__()
    self.backbone = backbone
    self.projection = tf.keras.layers.Dense(output_dim)

  def call(self, x):
    return self.projection(self.backbone(x)[1])

  def get_config(self):
    return super().get_config()


def get_loss(task_type: str):
  if task_type == 'multi_class':
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  elif task_type == 'multi_label':
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)

  else:
    raise ValueError


def get_optimizer(
    optimizer_name: str,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    weight_decay: float = 0.00001) -> tf.keras.optimizers.Optimizer:
  """Initialize optimizer by its name."""

  optimizer_name = optimizer_name.lower()
  if optimizer_name == 'adam':
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)
  elif optimizer_name == 'adamw':
    return tfa_optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay)
  elif optimizer_name == 'rmsprop':
    return tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate, momentum=momentum)
  elif optimizer_name == 'momentum':
    return tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay)
  elif optimizer_name == 'sgd':
    return tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0, weight_decay=weight_decay)
  else:
    raise ValueError('Optimizer %s not supported' % optimizer_name)


class MyF1Score(tfa_metrics.F1Score):
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.squeeze(tf.one_hot(y_true, self.num_classes), 1)
    super().update_state(y_true, y_pred, sample_weight)


def get_metrics(task_type: str, num_classes: int):
  if task_type == 'multi_class':
    return [MyF1Score(num_classes=num_classes,
                      average='micro',
                      name='micro_f1'),
            MyF1Score(num_classes=num_classes,
                      average='macro',
                      name='macro_f1'),
            tf.keras.metrics.SparseCategoricalAccuracy()]
  elif task_type == 'multi_label':
    return [tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(curve='PR',
                                 multi_label=True,
                                 num_labels=num_classes,
                                 from_logits=True,
                                 name='pr_auc')]
  else:
    raise ValueError


def get_early_stopping_callback(
    monitor_metric='val_loss',
    start_from_epoch=20,
    patience=5,
    verbose=1,
    mode='auto',
    restore_best_weights=True):

  return tf.keras.callbacks.EarlyStopping(
      monitor=monitor_metric,
      start_from_epoch=start_from_epoch,
      patience=patience,
      verbose=verbose,
      mode=mode,
      restore_best_weights=restore_best_weights,
  )


def get_checkpoint_callback(
    exp_dir,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch',
):
  checkpoint_string = '/checkpoints/epoch_{epoch:02d}'
  return tf.keras.callbacks.ModelCheckpoint(
      exp_dir + checkpoint_string,
      monitor=monitor,
      verbose=verbose,
      save_best_only=save_best_only,
      save_weights_only=save_weights_only,
      mode=mode,
      save_freq=save_freq,
  )


def get_tensorboard_callback(exp_dir):
  return tf.keras.callbacks.TensorBoard(
      log_dir=exp_dir + '/logs',
      write_graph=False,
      write_steps_per_second=True,
      update_freq='epoch')


def get_reduce_lr_plateau_callback(
    monitor='val_loss',
    factor=0.3,
    patience=10,
    verbose=1,
    mode='auto',
    min_lr=1e-5,
):
  return tf.keras.callbacks.ReduceLROnPlateau(
      monitor=monitor,
      factor=factor,
      patience=patience,
      verbose=verbose,
      mode=mode,
      min_lr=min_lr,
  )


def get_learning_rate_step_scheduler_callback(
    learning_rate=1e-4,
    factor=0.3,
    milestones=[30],
    verbose=1,
):
  def scheduler(epoch, learning_rate):
    if epoch in milestones:
      return learning_rate * factor
    else:
      return learning_rate

  return tf.keras.callbacks.LearningRateScheduler(
      scheduler,
      verbose=verbose,
  )


def get_callbacks(callbacks_names, exp_dir, monitor_metric, learning_rate):
  callbacks = []
  if 'checkpoint' in callbacks_names:
    callbacks.append(get_checkpoint_callback(exp_dir, monitor_metric))
  if 'reduce_lr_plateau' in callbacks_names:
    callbacks.append(get_reduce_lr_plateau_callback(monitor_metric))
  if 'step_scheduler' in callbacks_names:
    callbacks.append(get_learning_rate_step_scheduler_callback(
        learning_rate=learning_rate))
  if 'early_stopping' in callbacks_names:
    callbacks.append(get_early_stopping_callback())
  if 'tensorboard' in callbacks_names:
    callbacks.append(get_tensorboard_callback(exp_dir))
  return callbacks