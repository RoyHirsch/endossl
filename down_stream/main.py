"""Main script for linear evaluation / fine-tuning."""

import argparse
import config
import experiment


parser = argparse.ArgumentParser()

# Optional flags, to be used for parameters sweep
parser.add_argument(
  '--model',
  type=str,
  required=False)

parser.add_argument(
  '--optimizer',
  type=str,
  required=False)

parser.add_argument(
  '--learning_rate',
  type=float,
  required=False)

parser.add_argument(
  '--weight_decay',
  type=float,
  required=False)


def main(args):
    conf = config.Config()
    for k, v in vars(args).items():
        if v:
            setattr(conf, k, v)
    _ = experiment.run_experiment(conf)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)