"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/media/test/Data/cityscapes/data --subset=train --weights=coco

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
from glob import glob
import random
import json
track = 'E'
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print(ROOT_DIR)
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/alplab/")
with open('/media/test/Data/alplab/train.json') as json_file:
    train_examples = json.load(json_file)

with open('/media/test/Data/alplab/val.json') as json_file:
    val_examples = json.load(json_file)
'''
data_dir = '/media/test/Data/alplab/Track%s' % track

glob_path = os.path.join(data_dir, 'instancesNew/**', '*_IIMG.png')
iimg_files = glob(glob_path)

iimg_list_E = sorted([[int(i.split('Sphere-')[1].split('_')[0].split('-')[0]), i] for i in iimg_files],
                 key=lambda x: int(x[0]))
track = 'D'
data_dir = '/media/test/Data/alplab/Track%s' % track

glob_path = os.path.join(data_dir, 'instancesNew/**', '*_IIMG.png')
iimg_files = glob(glob_path)

iimg_list_D = sorted([[int(i.split('Sphere-')[1].split('_')[0].split('-')[0]), i] for i in iimg_files],
                 key=lambda x: int(x[0]))

iimg_list = iimg_list_E + iimg_list_D
# create bins for groups with consecutive id's (avoid mixing similar images between train and val)
id_bins = [[]]
id_bin_idx = 0
next_id = min([i[0] for i in iimg_list])
for i in iimg_list:
# consider a few skipped images still in consecutive group
    if next_id + 8 >= i[0]:
      id_bins[id_bin_idx].append(i)
    else:
      id_bins.append([i])
      id_bin_idx += 1

    next_id = i[0] + 1
random.seed(42)
random.shuffle(id_bins)
id_bins_cumsum = np.array([len(i) for i in id_bins]).cumsum()

num_examples = len(iimg_list)
num_train = int(0.70 * num_examples)
id_bin_idx = np.where([id_bins_cumsum >= num_train])[1][0]

train_examples = [e[1] for ix in range(id_bin_idx + 1) for e in id_bins[ix]]
val_examples = [e[1] for ix in range(id_bin_idx + 1, len(id_bins)) for e in id_bins[ix]]

num_train = len(train_examples)

random.shuffle(train_examples)
random.shuffle(val_examples)

with open('/media/test/Data/alplab/train.json', 'w') as outfile:
    json.dump(train_examples, outfile)

with open('/media/test/Data/alplab/val.json', 'w') as outfile:
    json.dump(val_examples, outfile) '''

############################################################
#  Configurations
############################################################

class TrafficConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "alplab"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus
    IMAGE_CHANNEL_COUNT =  5

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 400
    VALIDATION_STEPS = max(1, len(val_examples) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22, 40.54, 45.02 ])


class InferenceConfig(TrafficConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()



############################################################
#  Dataset
############################################################

class TrafficDataset(utils.Dataset):

    def load_traffic(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("alplab", 1, "traffic sign")
        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val"]
        if subset == "val":
            image_ids = val_examples

        if subset == "train":
            image_ids = train_examples

        # Add images
        for i, image_id in enumerate(image_ids):
            js = image_id.split('instancesNew/cam')
            cam_no = js[1][0]
            name = js[1].split('-cam%s_IIMG.' % cam_no)[0][2:]
            img_path = '/'.join([js[0] + 'RGBDI_img', 'cam' + cam_no, name + '-cam%s.tif' % cam_no])
            mask_path = image_id
            self.add_image(
                "alplab",
                image_id=i,
                path=img_path, mask=mask_path)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        info = self.image_info[image_id]
        img_path = info['path']
        image = skimage.io.imread(img_path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_fp = info['mask']
        m = skimage.io.imread(mask_fp).astype(np.bool)
        mask = []
        mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "alplab":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = TrafficDataset()
    dataset_train.load_traffic(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TrafficDataset()
    dataset_val.load_traffic(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=80,
                augmentation=augmentation,
                layers='all')


############################################################
#  Evaluation
############################################################

def evaluate(model, dataset_dir):
    f = open("rgbdi_mAP.txt", "a+")
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = val_examples
    APs = []
    dataset_val = TrafficDataset()
    dataset_val.load_traffic(dataset_dir, "val")
    dataset_val.prepare()
    for i, image_id in enumerate(image_ids):
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config, i, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        print(image)
        print(i)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        print(r)
        f.write(str(r))
        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        print('AP: %f' %AP)
        APs.append(AP)
        f.write(str(AP))

    print("mAP: ", np.mean(APs))
    f.write("mAP: %f" % np.mean(APs))


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
   

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TrafficConfig()
        config.display()
    else:
        config = InferenceConfig()
        config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
   
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask", "conv1"])
    else:
        model.load_weights(weights_path, by_name=True,  exclude=["conv1"])

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)

    elif args.command == "val":
        evaluate(model, args.dataset)

    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
