import librosa
from librosa.util import find_files
import numpy as np
import os
from pathlib import Path

import tensorflow as tf

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(wave_mix, wave_vocal, wave_bass, wave_drums, wave_other):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'mix': _bytes_feature(wave_mix),
      'vocal': _bytes_feature(wave_vocal),
      'bass': _bytes_feature(wave_bass),
      'drums': _bytes_feature(wave_drums),
      'other': _bytes_feature(wave_other),
  }

  # Create a Features message using tf.train.Example.
  return tf.train.Example(features=tf.train.Features(feature=feature))

# def open_file_and_convert_byte(filename):

def make_tfrecords(directory, tfrecords_name):
    record_file = f'{tfrecords_name}.tfrecords'
    print(os.path.join(directory, 'Spectrogram'))
    assert os.path.isdir(os.path.join(directory, 'Spectrogram'))
    filelist = find_files(os.path.join(directory, 'Spectrogram'), ext="npz")
    with tf.io.TFRecordWriter(record_file) as writer:
        # print(f"filelist len : {len(filelist)}")
        for file in filelist:
            data = np.load(file)
            mag_mix, _ = librosa.magphase(data['mix'])
            mag_vocal, _ = librosa.magphase(data['vocal'])
            mag_bass, _ = librosa.magphase(data['bass'])
            mag_drums, _ = librosa.magphase(data['drums'])
            mag_other, _ = librosa.magphase(data['other'])

            mag_mix = bytes(mag_mix)
            mag_vocal = bytes(mag_mix)
            mag_bass = bytes(mag_bass)
            mag_drums = bytes(mag_drums)
            mag_other = bytes(mag_other)

            # Write the raw image files to tfrecords files.
            # First, process the two images into `tf.Example` messages.
            # Then, write to a `.tfrecords` file.
            tf_example = serialize_example(mag_mix, mag_vocal, mag_bass, mag_drums, mag_other)
            writer.write(tf_example.SerializeToString())

def load_tfrecords(tfrecords_path):
    assert os.path.isfile(tfrecords_path)
    assert os.path.splitext(tfrecords_path)[1] == ".tfrecords"

    raw_dataset = tf.data.TFRecordDataset(tfrecords_path)

    feature_description = {
        'mix': tf.io.FixedLenFeature([], tf.string),
        'vocal': tf.io.FixedLenFeature([], tf.string),
        'bass': tf.io.FixedLenFeature([], tf.string),
        'drums': tf.io.FixedLenFeature([], tf.string),
        'other': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_wave_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    return raw_dataset.map(_parse_wave_function)

if __name__ == "__main__" :
    # make_tfrecords(".", "sound_seperator")
    dataset = load_tfrecords('sound_seperator.tfrecords')

for idx, feature in enumerate(dataset):
    # print(feature['bass'].numpy())
    # result = np.frombuffer(feature['bass'].numpy(), dtype=np.float32)
    result = tf.io.decode_raw(feature['bass'], out_type=tf.float32)
    print(f"len(result) : {len(result)}")
    print(f"result : {result}")
    print(f"type(result) : {type(result)}")
    if idx == 0:
        break

# def save_tfrecord(self, name: str, func, shards: int = 1):
#     filename = f'{self.path}/{self.name}-{name}.tfrecord'
#     with tf.io.TFRecordWriter(filename) as writer:
#         for count, (data, max_img, max_l) in enumerate(func):
#             data = self._serialize_data(data)
#             writer.write(data)
#
#     data = tf.data.TFRecordDataset(filename)
#     for i in range(shards):
#         writer = tf.data.experimental.TFRecordWriter(
#             f'{filename}-{i:05}-of-{shards:05}')
#         writer.write(data.shard(shards, i))
#
#     self.info.split(name, shards, count+1, max_img, max_l)
#     Path(filename).unlink()
