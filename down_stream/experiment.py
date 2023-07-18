"""A compact module for running down-stream experiments."""

import sys
import os
sys.path.append(os.path.realpath(__file__+ '/../../'))

import tensorflow as tf

from data import cholec80_images
from train import eval_lib
from train import train_lib


def verbose_print(msg, verbose=True, is_title=False):
  if verbose:
    print(msg)
  if is_title:
      print('#' * 40)


def run_experiment(config, verbose=True):
  """Stand-alone function for running an experiment."""

  verbose_print('Config:', verbose)
  attr_names = [i for i in dir(config) if not i.startswith('__')]
  for a in attr_names:
    verbose_print('{} = {}'.format(a, getattr(config, a, None)), verbose)
  print('\n\n')

  if not os.path.exists(config.exp_dir):
    os.makedirs(config.exp_dir)

  ##############################################################################
  # Datasets & Model
  ##############################################################################
  verbose_print('Create datasets and model', verbose, True)
  datasets = cholec80_images.get_cholec80_images_datasets(
      data_root=config.data_root,
      batch_size=config.batch_size,
      train_transformation=config.train_transformation,
  )


  if config.is_linear_evaluation:
    model = train_lib.get_linear_model(
        input_dim=config.input_dim, output_dim=config.num_classes
    )
  elif config.model == 'resnet50':
    input_tensor = tf.keras.Input(shape=(224, 224, 3,))
    backbone = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor)
    out = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    out = tf.keras.layers.Dense(config.num_classes)(out)
    model = tf.keras.Model(input_tensor, out, name='Model')
  elif 'vit' in config.model:
    backbone = tf.saved_model.load(config.saved_model_dir)
    model = train_lib.LinearFineTuneModel(backbone, config.num_classes)
  else:
    raise ValueError('Invalid model name: {}'.format(config.model))

  model.compile(
      optimizer=train_lib.get_optimizer(
          config.optimize_name,
          config.learning_rate,
          config.momentum,
          config.weight_decay,
      ),
      loss=train_lib.get_loss(config.task_type),
      metrics=train_lib.get_metrics(config.task_type, config.num_classes),
  )

  ##############################################################################
  # Train
  ##############################################################################
  verbose_print('Begin training', verbose, True)
  history = model.fit(
      datasets['train'],
      batch_size=config.batch_size,
      epochs=config.num_epochs,
      validation_data=datasets['validation'],
      class_weight=(cholec80_images._CHOLEC80_PHASES_WEIGHTS 
                    if config.use_class_weight else None),
      callbacks=train_lib.get_callbacks(
          callbacks_names=config.callbacks_names,
          exp_dir=config.exp_dir,
          monitor_metric=config.monitor_metric,
          learning_rate=config.learning_rate,
      ),
      validation_freq=config.validation_freq,
  )

  #############################################################################
  # End of train evaluation
  #############################################################################
  if config.manually_load_best_checkpoint:
    checkpoints = os.listdir(os.path.join(config.exp_dir, 'checkpoints') + '/*')
    if checkpoints:
      latest = checkpoints[-1]
      print(f'Load latest checkpoint: {latest}')
      model.load_weights(latest)
    else:
      print('Haven\'t loaded a saved checkpoint')

  # For the special case of phases, re-extract the dataset with the 'with_image_path'
  # attribute for calculating video-level metrics
  verbose_print('Start end of train evaluation', verbose, True)
  datasets = cholec80_images.get_cholec80_images_datasets(
      data_root=config.data_root,
      batch_size=config.batch_size,
      train_transformation=config.train_transformation,
      with_image_path=True,
  )

  mets = eval_lib.end_of_training_evaluation(
      model,
      datasets['validation'],
      datasets['test'],
      label_key=config.label_key,
      exp_dir=config.exp_dir,
      epoch=config.num_epochs)
  return mets, history