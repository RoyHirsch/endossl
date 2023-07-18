"""Visualization, monitoring and logging utils."""

import matplotlib.pyplot as plt


def plot_history(history, mets=['auc_10'], figsize=(15, 3)):
  """Plot keras.fit history graphs."""

  loss_list = [
      s for s in history.history.keys() if 'loss' in s and 'val' not in s
  ]
  val_loss_list = [
      s for s in history.history.keys() if 'loss' in s and 'val' in s
  ]

  epochs = range(1, len(history.history[loss_list[0]]) + 1)
  plt.figure(figsize=figsize)
  ax = plt.subplot(1, 1 + len(mets), 1)

  for l in loss_list:
    ax.plot(
        epochs,
        history.history[l],
        'b',
        label='Train loss ('
        + str(str(format(history.history[l][-1], '.3f')) + ')'),
    )
  for l in val_loss_list:
    ax.plot(
        epochs,
        history.history[l],
        'g',
        label='Valid loss ('
        + str(str(format(history.history[l][-1], '.3f')) + ')'),
    )

  ax.set_title('Loss')
  ax.set_xlabel('Epochs')
  ax.set_ylabel('Loss')
  ax.legend()

  for i, met_name in enumerate(mets):
    ax = plt.subplot(1, 1 + len(mets), 2 + i)

    if met_name == 'lr':
      ax.plot(epochs, history.history['lr'], 'b')

    else:
      train_met_name = [
          s for s in history.history.keys() if met_name in s and 'val' not in s
      ][0]
      val_met_name = [
          s for s in history.history.keys() if met_name in s and 'val' in s
      ][0]

      ax.plot(
          epochs,
          history.history[train_met_name],
          'b',
          label=f'Train {met_name} ('
          + str(str(format(history.history[train_met_name][-1], '.3f')) + ')'),
      )
      ax.plot(
          epochs,
          history.history[val_met_name],
          'g',
          label=f'Valid {met_name} ('
          + str(str(format(history.history[val_met_name][-1], '.3f')) + ')'),
      )

    ax.set_title(met_name)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(met_name)
    ax.legend()