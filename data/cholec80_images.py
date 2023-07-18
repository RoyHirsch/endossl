'''Module for creating TF datasets for Cholec80 dataset'''

import os
import tensorflow as tf
import tensorflow_models as tfm


_CHOLEC80_PHASES_WEIGHTS = {
    0: 1.9219914802981897,
    1: 0.19571110990619747,
    2: 0.9849911311229362,
    3: 0.2993075998175712,
    4: 1.942680301399354,
    5: 1.0,
    6: 2.2015858493443123
    }


_LABEL_NUM_MAPPING = {
    'GallbladderPackaging': 0,
    'CleaningCoagulation': 1,
    'GallbladderDissection': 2,
    'GallbladderRetraction': 3,
    'Preparation': 4,
    'ClippingCutting': 5,
    'CalotTriangleDissection': 6
    }


_SUBSAMPLE_RATE = 25


_CHOLEC80_SPLIT = {'train': range(1, 41),
                   'validation': range(41, 49),
                   'test': range(49, 81)}


curr_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(curr_dir, 'config.json')


resize = tf.keras.layers.Resizing(224, 224, crop_to_aspect_ratio=False)


resize_and_center_crop = tf.keras.layers.Resizing(
    224, 224, crop_to_aspect_ratio=True
)


_RAND_AUGMENT = tfm.vision.augment.RandAugment(
    num_layers=3, magnitude=7, exclude_ops=['Invert', 'Solarize', 'Posterize']
)


def randaug(image):
  image = resize(image)
  return _RAND_AUGMENT.distort(image * 255.0) / 255.0


def get_train_image_transformation(name):
  if name == 'randaug':
    return randaug
  else:
    return resize


class Cholec80ImagesLoader:
    def __init__(self, data_root, video_ids, batch_size, shuffle=False, augment=resize):
        self.batch_size = batch_size
        self.video_ids = video_ids
        self.data_root = data_root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        self.all_frame_names, self.all_labels  = self.prebuild(video_ids)

    def prebuild(self, video_ids):
        frames_dir = os.path.join(self.data_root, 'frames')
        annos_dir = os.path.join(self.data_root, 'phase_annotations')

        all_labels = []
        all_frame_names = []
        for video_id in video_ids:
            video_frames_dir = os.path.join(frames_dir, video_id)
            frames = [os.path.join(video_frames_dir, f) for f in os.listdir(video_frames_dir)]
            with open(os.path.join(annos_dir, video_id + '-phase.txt'), 'r') as f:
                labels = f.readlines()[1:]
            labels = [l.split('\t')[1][:-1] for l in labels]
            labels = [_LABEL_NUM_MAPPING[l] for l in labels[::_SUBSAMPLE_RATE]][:len(frames)]
            all_frame_names += frames
            all_labels += labels
        return all_frame_names, all_labels
    
    def parse_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        return self.augment(img)

    def parse_label(self, label):
        return label

    def parse_example(self, image_path, label):
        return (self.parse_image(image_path),
                self.parse_label(label))
        
    def parse_example_image_path(self, image_path, label, image_path_):
        return (self.parse_image(image_path),
                self.parse_label(label),
                image_path_)

    def get_tf_dataset(self, with_image_path=False):
        num_parallel_calls=tf.data.AUTOTUNE

        ds_frames = tf.data.Dataset.list_files(self.all_frame_names, shuffle=False)
        ds_labels = tf.data.Dataset.from_tensor_slices(self.all_labels)
        if with_image_path:
            ds = tf.data.Dataset.zip((ds_frames, ds_labels, ds_frames))
            ds = ds.map(self.parse_example_image_path, num_parallel_calls=num_parallel_calls)
        else:
            ds = tf.data.Dataset.zip((ds_frames, ds_labels))
            ds = ds.map(self.parse_example, num_parallel_calls=num_parallel_calls)

        ds = ds.batch(self.batch_size)
        if self.shuffle:
            ds = ds.shuffle(1024)
        ds = ds.prefetch(num_parallel_calls)
        return ds


def get_cholec80_images_datasets(data_root, batch_size, train_transformation='randaug', with_image_path=False):
    data = {}
    for split, ids_range in _CHOLEC80_SPLIT.items():
        if split == 'train':
            ds = Cholec80ImagesLoader(
                data_root,
                [f'video{i:02}' for i in ids_range],
                batch_size,
                augment=get_train_image_transformation(train_transformation)
                )
        data[split] = ds.get_tf_dataset(with_image_path)
    return data


if __name__ == '__main__':
    par_dir = os.path.realpath(__file__+ '/../../')
    data_root = os.path.join(par_dir, 'cholec80', 'cholec80')
    datasets = get_cholec80_images_datasets(data_root, 8)
    for b in datasets['validation']:
        print(b[0].shape)
        print(b[1].shape)
        break