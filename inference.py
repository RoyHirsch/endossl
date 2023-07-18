'''Simple script for loading pre-trained model and extracting hidden representations from it.'''

import tensorflow as tf
import argparse


def get_image(image_path, size=224):
    img = tf.keras.preprocessing.image.load_img(image_path)
    img = tf.convert_to_tensor(img)
    return tf.image.resize(img, [size, size]) 


class ModelWrapper(tf.keras.Model):
  def __init__(self, model_path):
    super().__init__()
    self.backbone = tf.saved_model.load(model_path)

  def call(self, x):
    return self.backbone(x)[1]

  def get_config(self):
    return super().get_config()


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path",
    type=str,
    default='/home/royhirsch/research-il-lapmsn/checkpoints/vits_lapro_private/saved_model_inference',
    help="Path to a pre-trained model",
)
parser.add_argument(
    "--image_path",
    type=str,
    default='/home/royhirsch/research-il-lapmsn/cholec80/samples/video01_000001.png',
    help="Path to a image file",
)

def main(args):
    model = ModelWrapper(args.model_path)
    img = get_image(args.image_path)
    embed = tf.squeeze(model(tf.expend_dims(img, 0)), 0)
    print('Hidden vector shape is: {}'.format(embed.shape))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)