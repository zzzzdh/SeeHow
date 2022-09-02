from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import re
sys.path.append('/home/cheer/Project/ActionNet/lib')
import skimage
from feature_extractor.feature_extractor import FeatureExtractor
from tqdm import tqdm

data_dir = '/home/cheer/Project/Workflow/data'
ck_path = '/home/cheer/Project/ActionNet/models/action_vgg_e/0/jp/model.ckpt-10000'

nums = 19
batch_size = 1
net_name = 'action_vgg_e'
input_mode = 2
output_mode = 0

def classification_placeholder_input(feature_extractor, image_a, image_b, logits_name, batch_size, num_classes):

  batch_image1 = np.zeros([batch_size, feature_extractor.image_size, feature_extractor.image_size, 3], dtype=np.float32)
  batch_image2 = np.zeros([batch_size, feature_extractor.image_size, feature_extractor.image_size, 3], dtype=np.float32)

  for i in range(batch_size):
    channels = np.split(image_a, 3, axis=2)
    red = channels[0]
    green = channels[1]
    blue = channels[2]
    channels[0] = blue - 103.94
    channels[1] = green - 116.78
    channels[2] = red - 123.68
    image_a = np.concatenate((channels[0],channels[1],channels[2]),axis=2)
    batch_image1[i] = image_a

  for i in range(batch_size):
    channels = np.split(image_b, 3, axis=2)
    red = channels[0]
    green = channels[1]
    blue = channels[2]
    channels[0] = blue - 103.94
    channels[1] = green - 116.78
    channels[2] = red - 123.68
    image_b = np.concatenate((channels[0],channels[1],channels[2]),axis=2)

    batch_image2[i] = image_b

  outputs = feature_extractor.feed_forward_batch([logits_name], batch_image1, batch_image2, fetch_images=True)

  predictions = outputs[logits_name]  

  if output_mode == 2:
    viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(predictions, transitions)
    predictions = viterbi_sequence
  else:
    predictions = np.argmax(predictions, axis=1)
  #print(predictions)
  return predictions

def main():
  feature_extractor = FeatureExtractor(
    network_name=net_name,
    input_mode = input_mode,
    output_mode = output_mode,
    checkpoint_path=ck_path,
    batch_size=batch_size,
    num_classes=nums,
    preproc_func_name=net_name)
  feature_extractor.print_network_summary()

  annotations = os.listdir(os.path.join(data_dir, 'Annotations'))
  actions = os.listdir(os.path.join(data_dir, 'Actions'))
  for action in actions:
    if action in annotations:
      annotations.remove(action)
      print ('remove ' + action)
  for annotation in annotations:
    print (annotation)
    with open(os.path.join(data_dir, 'Annotations', annotation), 'r') as annotation_file:
      annotation_list = annotation_file.readlines()
    image1 = image = skimage.io.imread(os.path.join(data_dir, 'Images', os.path.splitext(annotation)[0], annotation_list[0].split()[0] + '.jpg'))
    action_list = []
    for line in tqdm(annotation_list):
      image_name = line.split()[0] + '.jpg'
      box = [int(x) for x in line.split()[1:5]]
      image2 = skimage.io.imread(os.path.join(data_dir, 'Images', os.path.splitext(annotation)[0], image_name))
      if sum(box):
        diff = (box[3] - box[1]) * (box[2] - box[0]) / (image2.shape[0] * image2.shape[1])
        image1_clip = image1[box[1]:box[3], box[0]:box[2], :]
        image2_clip = image2[box[1]:box[3], box[0]:box[2], :]
        image1_clip = skimage.transform.resize(image1_clip, (224, 224))
        image2_clip = skimage.transform.resize(image2_clip, (224, 224))
        clip_class = classification_placeholder_input(feature_extractor, skimage.img_as_ubyte(image1_clip), skimage.img_as_ubyte(image2_clip), net_name + '/fc8',batch_size, 19)
        clip_class = clip_class[0] if clip_class[0] in range(6, 15) else 0
      else:
        clip_class = 0
      action_list.append(line.split()[0] + ' ' + str(clip_class) + '\n')
      image1 = image2.copy()
    with open(os.path.join(data_dir, 'Actions', annotation), 'w') as action_file:
      action_file.writelines(action_list)

if __name__ == '__main__':
  main()
