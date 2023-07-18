"""Base config class."""

import os


par_dir = os.path.realpath(__file__+ '/../../')


class Config:
  """Base config class."""

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

  exp_dir = os.path.join(par_dir, 'exps', 'tmp')

  # dataset
  dataset_name = 'cholec80'
  data_root = os.path.join(par_dir, 'cholec80', 'cholec80')
  train_transformation = 'randaug'
  label_key = 'segment'

  is_linear_evaluation = False
  model = 'resnet50'
  saved_model_dir = os.path.join(par_dir, 'checkpoints', 'pretrained_model_dir')
  task_type = 'multi_class'
  monitor_metric = 'val_macro_f1'
  input_dim = {'vits': 384, 'vitb': 768, 'vitl': 1024, 'resnet50': 2048}[model]
  num_classes = 7

  # optimization
  use_class_weight = False
  optimize_name = 'adamw'
  learning_rate = 1e-4
  momentum = 0.9
  weight_decay = 1e-5

  num_epochs = 5
  batch_size = 256
  validation_freq = 1
  callbacks_names = ['checkpoint', 'reduce_lr_plateau', 'early_stopping']
  manually_load_best_checkpoint = True