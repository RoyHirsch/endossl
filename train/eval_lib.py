"""Evaluation methods."""

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import tensorflow as tf


def calc_f1(model, ds, agg='video', verbose=0):
  video2labels = {}
  video2preds = {}

  all_labels = []
  all_preds = []
  for batch in ds:

    inputs, labels, clip_paths = batch
    preds = model.predict(inputs, verbose=verbose)
    preds = tf.argmax(preds, 1)

    all_labels += labels.numpy().tolist()
    all_preds += preds.numpy().tolist()

    if agg == 'video':
      for c, l, p in zip(clip_paths.numpy(), labels.numpy(), preds.numpy()):
        c = c.decode('utf-8').split('/')[-2]
        if c not in video2labels:
          video2labels[c] = []
          video2preds[c] = []
        video2labels[c].append(l)
        video2preds[c].append(p)

  ##################
  # Summarize
  ##################
  # anyway calculate frame-level metrics
  all_labels = np.asarray(all_labels)
  all_preds = np.asarray(all_preds)
  frame_mets = classification_report(all_labels, all_preds, output_dict=True)

  if agg == 'frame':
    all_labels = np.asarray(all_labels)
    all_preds = np.asarray(all_preds)

    return {
        'acc': np.round(
            np.sum(all_labels == all_preds) * 100 / len(all_labels), 2
        ),
        'f1': np.round(score(all_labels, all_preds)[-2].mean() * 100, 2),
    }

  elif agg == 'video':
    accs = []
    scores = []
    for sub_labels, sub_preds in zip(
        video2labels.values(), video2preds.values()
    ):
      sub_labels = np.asarray(sub_labels)
      sub_preds = np.asarray(sub_preds)

      # compute acc and append
      vid_acc = np.sum(sub_labels == sub_preds) * 100 / len(sub_labels)
      accs.append(vid_acc)

      # compute F1
      vid_score = score(sub_labels, sub_preds)
      mean = np.mean(np.vstack(vid_score).T, axis=0)
      mean[:-1] *= 100
      scores.append(mean)

    # summarize
    overall_acc = np.around(np.mean(np.stack(accs)), 2)
    overall_f1 = np.mean(np.stack(scores), axis=0)
    overall_f1 = np.around(overall_f1, 2)

    return {
        'video_acc': overall_acc,
        'video_precision': overall_f1[0],
        'video_recall': overall_f1[1],
        'video_f1': overall_f1[2],

        'frame_acc': np.around(
            frame_mets['accuracy'] * 100, 2),
        'frame_macro_f1': np.around(
            frame_mets['macro avg']['f1-score'] * 100, 2),
        'frame_micro_f1': np.around(
            frame_mets['weighted avg']['f1-score'] * 100, 2),
        }


def calc_map(model, ds, agg='all', verbose=0):

  all_labels = np.empty(())
  all_preds = np.empty(())
  for i, batch in enumerate(ds):
    inputs, labels = batch
    preds = model.predict(inputs, verbose=verbose)
    if i == 0:
      all_labels = labels.numpy()
      all_preds = preds
    else:
      all_labels = np.concatenate((all_labels, labels.numpy()), 0)
      all_preds = np.concatenate((all_preds, preds), 0)
  try:
    mean = [mean_ap(all_labels, all_preds) * 100]
    std = [0.00]
    if agg == 'class':
      mean = mean_ap(all_labels, all_preds, mean=False) * 100
      std = [0.00] * 7
  except:
    mean = [-1] * 7 if agg == 'class' else [-1.00]
    std = [0.00] * 7 if agg == 'class' else [0.00]
  mean = [np.round(i, 2) for i in mean]
  std = [np.round(i, 2) for i in std]
  if len(mean) == 1:
    return {'mean': mean[0], 'std': std[0]}
  return {'mean': mean, 'std': std}


def mean_ap(labels, predictions, mean=True):
  metrics = np.array(average_precision_score(labels, predictions, average=None))
  if mean:
    metrics = np.sum([x for x in metrics]) / len(metrics)
  return metrics

def end_of_training_evaluation(
    model, validation_ds, test_ds, label_key, exp_dir, epoch):
  """Run aggregative evaluation over the trained model."""

  if label_key == 'tool':
    validation_map = calc_map(model, validation_ds)
    test_map = calc_map(model, test_ds)
    mets = {'val_map': validation_map['mean'], 'test_map': test_map['mean']}

  elif label_key == 'segment':
    mets = {}
    for k, v in calc_f1(model, validation_ds, agg='video').items():
      mets[f'val_{k}'] = v
    for k, v in calc_f1(model, test_ds, agg='video').items():
      mets[f'test_{k}'] = v

  else:
    raise ValueError

  file_writer = tf.summary.create_file_writer(exp_dir + '/metrics')
  file_writer.set_as_default()
  for k, v in mets.items():
    tf.summary.scalar(k, data=v, step=epoch)
  return mets
