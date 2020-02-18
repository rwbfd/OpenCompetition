#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 19:35:50 2018

@author: rudl
"""

#import json
import cv2
import random
import hashlib
import sys
import os
import io
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import numpy as np
import tensorflow as tf
import sys

import cv2
import hashlib
import io
import logging
import os
from glob import glob
import json
from shapely.geometry import Polygon, mapping
import random
import re

import contextlib2
import numpy as np
import PIL.Image
import PIL.ImageDraw
import tensorflow as tf
sys.path.append('/home/test/tensorflow/models/research/object_detection')
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
sys.path.append('/media/test/Data/cityscapes/cityscapesscripts/helpers')
from labels import labels
track = 'D'

from utils import dataset_util

class_mappings = {0: 'background', 1: 'traffic sign', 2: 'traffic light'}

if 'FLAGS' not in locals():
  flags = tf.app.flags
  flags.DEFINE_string('data_dir', '/media/test/Data/alplab/Track%s' % track, 'Root directory of the raw a9 dataset.')
  flags.DEFINE_string('output_dir', '/media/test/Data/alplab/Track%s/tfrecords' % track, 'Path to directory to output TFRecords.')
  flags.DEFINE_integer('max_img', None, 'Limit the total number of Images (size).')
  flags.DEFINE_float('perc_train', 0.75, 'Percentage of the avalable Examples to use for training.')
  flags.DEFINE_float('min_area', 0.00001, 'Minimum area of a box to be considered in the dataset. Area is in percentage of the image area, thus 1.0 would be a box covering the whole image.')
  flags.DEFINE_integer('splits_divider', 2, 'Divide the input image into this number of stripes [full heigth]')
  flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')
#  flags.DEFINE_integer('splits_add', 3, 'Crate additional random full heigth stripes with the with of imageWidth/splits_divider')
  flags.DEFINE_bool('debug', False, 'Show debug infos.')
  flags.DEFINE_bool('vmasks', False, 'Visualize masks')
  FLAGS = flags.FLAGS

DATA_DIR = os.path.abspath(FLAGS.data_dir)
OUTPUT_DIR = os.path.abspath(FLAGS.output_dir)

# Print an error message and quit
def printError(message):
  print('ERROR: {}'.format(message))
  print('')
  sys.exit(-1)
   
def sub_img_to_tf_example(img_name, image, instanceImg):
  
  encoded_image = io.BytesIO()
  image.save(encoded_image, format='JPEG')
  key = hashlib.sha256(encoded_image.getvalue()).hexdigest()

  iimg_np = np.asarray(instanceImg).copy()
  iimg_vals = np.unique(iimg_np)
  assert(len(iimg_vals) > 0 and iimg_vals[0] == 0)
  instances = iimg_vals[1:]
  
  #tf.logging.debug("%s values: %s" % (img_name, iimg_vals))
  
  #if FLAGS.debug:
    #tf.logging.log_every_n(tf.logging.INFO, "%s (%ix%i): %02i instances" % (img_name, imgWidth, imgHeight, num_instances), 100)

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  masks = []
  #tf.logging.log(tf.logging.DEBUG, '%i' % num_instances)
  for (i, j) in enumerate(instances):
    try:
      inst_class = 1 #we currently only have traffic signs
      
      mask_bin = (iimg_np == j)
      mask = mask_bin.astype(np.uint8) * 1 #now mask is 0 or 1

      output = io.BytesIO()
      #encode the mask as png
      mask_png = Image.fromarray(mask)
      mask_png.save(output, format='PNG')

      #calculate a box arround the mask
      indices_x = np.any(iimg_np == j, axis=0)
      indices_y = np.any(iimg_np == j, axis=1)
      x_mask = np.where(indices_x)
      y_mask = np.where(indices_y)
  
      xmin = np.min(x_mask)
      xmax = np.max(x_mask)
      ymin = np.min(y_mask)
      ymax = np.max(y_mask)

      x_fraction = (xmax-xmin)/image.width
      y_frcation = (ymax-ymin)/image.height
      area = x_fraction * y_frcation 
      
      if area < FLAGS.min_area:
        if area > 0:
          tf.logging.log(tf.logging.WARN, '%02i/%02i (%s) has area < treshold => %02.7f < %02.7f' % (i, inst_class, img_name, area, FLAGS.min_area))
        else:
          tf.logging.log(tf.logging.ERROR, '%02i/%02i (%s) has area < treshold => %02.7f < %02.7f' % (i, inst_class, img_name, area, FLAGS.min_area))
          
        continue

      #if FLAGS.debug:
       # mask_png.save(os.path.join(OUTPUT_DIR, '%s%02i_instances.png' % (img_name, i)))

      masks.append(output.getvalue())
      xmins.append(xmin.astype(np.float)/image.width)
      xmaxs.append(xmax.astype(np.float)/image.width)
      ymins.append(ymin.astype(np.float)/image.height) 
      ymaxs.append(ymax.astype(np.float)/image.height)
    
      classes.append(inst_class)
      #classes_text.append('traffic sign'.encode('utf8'))
      classes_text.append(class_mappings[inst_class].encode('utf8'))
    
      tf.logging.log(tf.logging.DEBUG, '%02i: (%04i,%04i), (%04i,%04i)' % (i, xmin, ymin, xmax, ymax))
      
    except ValueError:
#      if FLAGS.debug:
#        instanceImg.save(os.path.join(OUTPUT_DIR, '%s%02i_instances.png' % (img_name, i)))
#        mask_png.save(os.path.join(OUTPUT_DIR, '%s%02i_single_mask.png' % (img_name, i)))
      tf.logging.warn("%s (%ix%i): %02i instances/#%02i having invalid mask (not an instance):\nx-vals: %s \ny-vals: %s\n" % (img_name, image.width, image.heigth, len(instances) - 1, i, x_mask, y_mask))
      continue

  #this image has no considerable boxes at all
  if len(classes) == 0:
    return None

  if FLAGS.debug:
    tf.logging.debug("%s: %02i instances used" % (img_name, len(masks)))
    #instanceImg.save(os.path.join(OUTPUT_DIR, '%s_%02i_instances.png' % (img_name, num_instances )))

  feature_dict = {
    'image/width': dataset_util.int64_feature(image.width),
    'image/height': dataset_util.int64_feature(image.height),
    'image/filename': dataset_util.bytes_feature(
        img_name.encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(
        img_name.encode('utf8')),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_image.getvalue()),
    'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
#    'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
#    'image/object/truncated': dataset_util.int64_list_feature(truncated),
#    'image/object/view': dataset_util.bytes_list_feature(poses),
    'image/object/mask': dataset_util.bytes_list_feature(masks)
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example

def dict_to_tf_examples(data):
  global DATA_DIR
  
  img_path = data['img_path']
  iimg_path = data['iimg_path']
  img_name = data['name']

  with tf.gfile.GFile(iimg_path, 'rb') as fid:
    image_file = fid.read()
    image_io = io.BytesIO(image_file)
    instanceImg = Image.open(image_io)

  with tf.gfile.GFile(img_path, 'rb') as fid:
    image_file = fid.read()
    image_io = io.BytesIO(image_file)
    image = Image.open(image_io)

  splits_divider = FLAGS.splits_divider  
  
  split_width = int(np.ceil(image.width/splits_divider))
  split_width_half = int(np.ceil(split_width/2))

  split_positions = [i*split_width for i in range(splits_divider-1)]
  split_positions.append(image.width - split_width)
  split_positions += [split_width_half + i*split_width for i in range(splits_divider-1)]
  #split_positions += [random.randint(10,(image.width-split_width-10)) for i in range(FLAGS.splits_add)]
  
  examples = []
  
  for i, pos in enumerate(split_positions):
    box = (pos, 0, pos+split_width, image.height)
    sub_image = image.crop(box)
    sub_instanceImg = instanceImg.crop(box)
    
    examples.append(sub_img_to_tf_example(img_name + '[#' + str(i) + ']', sub_image, sub_instanceImg))

  return(examples)
  
def create_tf_record(output_filename,
                     examples, num_shards):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    examples: Examples to parse and save to tf record.
  """



  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
      tf_record_close_stack, output_filename, num_shards)


    count = 0
    for i, j in enumerate(examples):
      tf.logging.log_every_n(tf.logging.INFO, 'On image %d of %d', 100, i, len(examples))

      js = j.split('instancesNew/cam')
      cam_no = js[1][0]
      name = js[1].split('-cam%s_IIMG.' % cam_no)[0][2:]

      data = {
        'iimg_path': j,
        'img_path': '/'.join([js[0] + 'GDI_img', 'cam' + cam_no, name + '-cam%s.jpg' % cam_no]),
        'name': cam_no + '_' + name
      }
      tf_examples = dict_to_tf_examples(data)
      for tf_example in tf_examples:
        if tf_example:

          shard_idx = i % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
          count += 1
    tf.logging.info('%i valid examples written', count)




def main(_):
  if FLAGS.debug:
    tf.logging.set_verbosity(tf.logging.DEBUG)
  else:
    tf.logging.set_verbosity(tf.logging.INFO)

  tf.logging.log(tf.logging.INFO, 'Reading from a9_Lassnitzhoehe dataset')
  tf.logging.log(tf.logging.INFO, 'Using database at : ''%s''' % DATA_DIR)
  tf.logging.log(tf.logging.INFO, 'Storing results in: ''%s''' % OUTPUT_DIR)
  tf.logging.log(tf.logging.INFO, '-' * 80)

  #many images have no traffic signs at all..
  #glob_path = os.path.join(DATA_DIR, 'gtFine/**/**', '*006153*polygons.json')
  glob_path = os.path.join(DATA_DIR, 'instancesNew/**', '*_IIMG.png')
  tf.logging.log(tf.logging.INFO, 'Searching for pattern >%s<' % glob_path)
  
  iimg_files = tf.gfile.Glob(glob_path)
  
  iimg_list = sorted([[int(i.split('Sphere-')[1].split('_')[0].split('-')[0]), i] for i in iimg_files], key=lambda x: int(x[0]))

  #create bins for groups with consecutive id's (avoid mixing similar images between train and val)
  id_bins = [[]]
  id_bin_idx = 0
  next_id =min([i[0] for i in iimg_list])
  for i in iimg_list:
    #consider a few skipped images still in consecutive group
    if next_id + 8 >= i[0]:
      id_bins[id_bin_idx].append(i)
    else:
      id_bins.append([i])
      id_bin_idx+= 1

    next_id = i[0] + 1
      
  tf.logging.log(tf.logging.INFO, 'Using %d annotation files...', len(iimg_list))

  random.seed(42)
  random.shuffle(id_bins)
  id_bins_cumsum = np.array([len(i) for i in id_bins]).cumsum()

  num_examples = len(iimg_list)
  num_train = int(FLAGS.perc_train * num_examples)
  id_bin_idx = np.where([id_bins_cumsum >= num_train])[1][0]

  train_examples = [e[1] for ix in range(id_bin_idx + 1) for e in id_bins[ix]]
  val_examples = [e[1] for ix in range(id_bin_idx+1,len(id_bins)) for e in id_bins[ix]]
  
  num_train = len(train_examples)
  
  random.shuffle(train_examples)
  random.shuffle(val_examples)

  train_output_path = os.path.join(OUTPUT_DIR, 'alplab_gdi_train.tfrecord')
  val_output_path = os.path.join(OUTPUT_DIR, 'alplab_gdi_val.tfrecord')
  tf.logging.log(tf.logging.INFO, 'Storing (%i+%i) examples as: \n%s\n%s\n', num_train, num_examples-num_train, train_output_path, val_output_path)

  create_tf_record(
      train_output_path,
      train_examples, FLAGS.num_shards)
  create_tf_record(
      val_output_path,
      val_examples, FLAGS.num_shards)

if __name__ == '__main__':
  tf.app.run()
