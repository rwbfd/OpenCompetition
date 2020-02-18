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

# cityscapes imports (https://github.com/mcordts/cityscapesScripts)
sys.path.append('/home/batz/contrib/cityscapesScripts/cityscapesscripts/helpers')
from annotation import Annotation
# labels _are modified_ for instances of street signs
from labels import labels, name2label

# tf_mopdels imports (https://https://github.com/tensorflow/models.git)
sys.path.append('/home/batz/contrib/tf_models/research/object_detection')
from utils import dataset_util

# from utils import label_map_util

class_mappings = {0: 'background', 1: 'traffic sign', 2: 'traffic light'}

if 'FLAGS' not in locals():
    flags = tf.app.flags
    flags.DEFINE_string('data_dir', '.', 'Root directory of the raw cityscapes dataset.')
    flags.DEFINE_string('output_dir', '.', 'Path to directory to output TFRecords.')
    flags.DEFINE_integer('max_img', None, 'Limit the total number of Images (size).')
    flags.DEFINE_float('perc_train', 0.9, 'Percentage of the avalable Examples to use for training.')
    flags.DEFINE_float('min_area', 0.00001,
                       'Minimum area of a box to be considered in the dataset. Area is in percentage of the image area, thus 1.0 would be a box covering the whole image.')
    flags.DEFINE_integer('splits_divider', 3, 'Divide the input image into this number of stripes [full heigth]')
    #  flags.DEFINE_integer('splits_add', 3, 'Crate additional random full heigth stripes with the with of imageWidth/splits_divider')
    flags.DEFINE_bool('debug', False, 'Show debug infos.')
    flags.DEFINE_bool('vmasks', False, 'Visualize masks')
    FLAGS = flags.FLAGS

DATA_DIR = os.path.abspath(FLAGS.data_dir)
OUTPUT_DIR = os.path.abspath(FLAGS.output_dir)


# Convert the given annotation to a label image
# this function is taken allmost 1:1 from cityscape scripts (https://github.com/mcordts/cityscapesScripts)
def createInstanceImage(annotation, encoding):
    # the size of the image
    size = (annotation.imgWidth, annotation.imgHeight)

    # the background
    if encoding == "ids":
        backgroundId = name2label['unlabeled'].id
    elif encoding == "trainIds":
        backgroundId = name2label['unlabeled'].trainId
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    instanceImg = Image.new("I", size, backgroundId)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw(instanceImg)

    # a dict where we keep track of the number of instances that
    # we already saw of each class
    nbInstances = {}
    for labelTuple in labels:
        if labelTuple.hasInstances:
            nbInstances[labelTuple.name] = 0

    # loop over all objects
    for obj in annotation.objects:
        label = obj.label
        polygon = obj.polygon

        # If the object is deleted, skip it
        if obj.deleted:
            continue

        # if the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        # also we know that this polygon describes a group
        isGroup = False
        if (not label in name2label) and label.endswith('group'):
            label = label[:-len('group')]
            isGroup = True

        if not label in name2label:
            printError("Label '{}' not known.".format(label))

        # the label tuple
        labelTuple = name2label[label]

        # get the class ID
        if encoding == "ids":
            id = labelTuple.id
        elif encoding == "trainIds":
            id = labelTuple.trainId

        # if this label distinguishs between individual instances,
        # make the id a instance ID
        if labelTuple.hasInstances and not isGroup and id != 255:
            id = id * 1000 + nbInstances[label]
            nbInstances[label] += 1

        # If the ID is negative that polygon should not be drawn
        if id < 0:
            continue

        try:
            drawer.polygon(polygon, fill=id)
        except:
            print("Failed to draw polygon with label {} and id {}: {}".format(label, id, polygon))
            raise

    return instanceImg


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
    assert (len(iimg_vals) > 0 and iimg_vals[0] == 0)
    instances = iimg_vals[1:]

    # tf.logging.debug("%s values: %s" % (img_name, iimg_vals))

    # if FLAGS.debug:
    # tf.logging.log_every_n(tf.logging.INFO, "%s (%ix%i): %02i instances" % (img_name, imgWidth, imgHeight, num_instances), 100)

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    masks = []
    # tf.logging.log(tf.logging.DEBUG, '%i' % num_instances)
    for (i, j) in enumerate(instances):
        try:
            # images are encoded (id * 1000) + instance
            inst_id = j % 1000
            inst_class = int((j - inst_id) / 1000)

            mask_bin = (iimg_np == j)
            mask = mask_bin.astype(np.uint8) * 2  # now mask is 0 or 2

            mask_first_pixel = tuple(np.column_stack(np.where(mask == 2))[0][::-1])

            # in allmost all cases this will just fill the single connected mask with ones
            cv2.floodFill(mask, None, mask_first_pixel, 1, flags=8 | cv2.FLOODFILL_FIXED_RANGE)

            # BUT in a few cases this will detect an additional, unconnected portion of the mask..
            # most probably poison
            if not np.alltrue(mask <= 1):
                tf.logging.log(tf.logging.WARN, '%02i/%02i (%s) has a split mask' % (i, inst_class, img_name))

                if FLAGS.vmasks:
                    cv2.imshow('image', mask * 255)
                    keyb = cv2.waitKey(0)

                    if keyb == 27:
                        sys.exit()

                    cv2.destroyAllWindows()

                    continue

            output = io.BytesIO()
            # encode the mask as png
            mask_png = Image.fromarray(mask)
            mask_png.save(output, format='PNG')

            # calculate a box arround the mask
            indices_x = np.any(iimg_np == j, axis=0)
            indices_y = np.any(iimg_np == j, axis=1)
            x_mask = np.where(indices_x)
            y_mask = np.where(indices_y)

            xmin = np.min(x_mask)
            xmax = np.max(x_mask)
            ymin = np.min(y_mask)
            ymax = np.max(y_mask)

            x_fraction = (xmax - xmin) / image.width
            y_frcation = (ymax - ymin) / image.height
            area = x_fraction * y_frcation

            if area < FLAGS.min_area:
                if area > 0:
                    tf.logging.log(tf.logging.WARN, '%02i/%02i (%s) has area < treshold => %02.7f < %02.7f' % (
                    i, inst_class, img_name, area, FLAGS.min_area))
                else:
                    tf.logging.log(tf.logging.ERROR, '%02i/%02i (%s) has area < treshold => %02.7f < %02.7f' % (
                    i, inst_class, img_name, area, FLAGS.min_area))

                continue

            # if FLAGS.debug:
            # mask_png.save(os.path.join(OUTPUT_DIR, '%s%02i_instances.png' % (img_name, i)))

            masks.append(output.getvalue())
            xmins.append(xmin.astype(np.float) / image.width)
            xmaxs.append(xmax.astype(np.float) / image.width)
            ymins.append(ymin.astype(np.float) / image.height)
            ymaxs.append(ymax.astype(np.float) / image.height)

            classes.append(inst_class)
            # classes_text.append('traffic sign'.encode('utf8'))
            classes_text.append(class_mappings[inst_class].encode('utf8'))

            tf.logging.log(tf.logging.DEBUG, '%02i: (%04i,%04i), (%04i,%04i)' % (i, xmin, ymin, xmax, ymax))

        except ValueError:
            #      if FLAGS.debug:
            #        instanceImg.save(os.path.join(OUTPUT_DIR, '%s%02i_instances.png' % (img_name, i)))
            #        mask_png.save(os.path.join(OUTPUT_DIR, '%s%02i_single_mask.png' % (img_name, i)))
            tf.logging.warn(
                "%s (%ix%i): %02i instances/#%02i having invalid mask (not an instance):\nx-vals: %s \ny-vals: %s\n" % (
                img_name, image.width, image.heigth, len(instances) - 1, i, x_mask, y_mask))
            continue

    # this image has no considerable boxes at all
    if len(classes) == 0:
        return None

    if FLAGS.debug:
        tf.logging.debug("%s: %02i instances used" % (img_name, len(masks)))
        # instanceImg.save(os.path.join(OUTPUT_DIR, '%s_%02i_instances.png' % (img_name, num_instances )))

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

    img_name = os.path.join(data['name'] + 'leftImg8bit.png')
    img_path = os.path.join(DATA_DIR, 'leftImg8bit', data['relpath'], img_name)

    annotation = Annotation()
    annotation.fromJsonFile(data['json_path'])
    instanceImg = createInstanceImage(annotation, "trainIds")
    instanceImg.Format = 'PNG'

    with tf.gfile.GFile(img_path, 'rb') as fid:
        image_file = fid.read()
        image_io = io.BytesIO(image_file)
        image = Image.open(image_io)

    splits_divider = FLAGS.splits_divider

    split_width = int(np.ceil(image.width / splits_divider))
    split_width_half = int(np.ceil(split_width / 2))

    split_positions = [i * split_width for i in range(splits_divider - 1)]
    split_positions.append(image.width - split_width)
    split_positions += [split_width_half + i * split_width for i in range(splits_divider - 1)]
    # split_positions += [random.randint(10,(image.width-split_width-10)) for i in range(FLAGS.splits_add)]

    examples = []

    for i, pos in enumerate(split_positions):
        box = (pos, 0, pos + split_width, image.height)
        sub_image = image.crop(box)
        sub_instanceImg = instanceImg.crop(box)

        examples.append(sub_img_to_tf_example(img_name + '[#' + str(i) + ']', sub_image, sub_instanceImg))

    return (examples)


def create_tf_record(output_filename,
                     examples):
    """Creates a TFRecord file from examples.

    Args:
      output_filename: Path to where output file is saved.
      examples: Examples to parse and save to tf record.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)

    count = 0
    for i, j in enumerate(examples):
        tf.logging.log_every_n(tf.logging.INFO, 'On image %d of %d', 100, i, len(examples))

        js = j.split('gtFine')[1].split('/')

        data = {
            'json_path': j,
            'relpath': '/'.join(js[1:3]),
            'name': js[3]
        }

        # tf.logging.log(tf.logging.DEBUG, data)

        #    try:
        tf_examples = dict_to_tf_examples(data)

        for tf_example in tf_examples:
            if tf_example != None:
                writer.write(tf_example.SerializeToString())
                count += 1

            if FLAGS.max_img != None and count > FLAGS.max_img:
                writer.close()
                tf.logging.info('[Stopped because] %i valid examples written', count)
                return

    writer.close()
    tf.logging.info('%i valid examples written', count)


def main(_):
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.log(tf.logging.INFO, 'Reading from Citiscapes dataset')
    tf.logging.log(tf.logging.INFO, 'Using database at : ''%s''' % DATA_DIR)
    tf.logging.log(tf.logging.INFO, 'Storing results in: ''%s''' % OUTPUT_DIR)
    tf.logging.log(tf.logging.INFO, '-' * 80)

    # many images have no traffic signs at all..
    # glob_path = os.path.join(DATA_DIR, 'gtFine/**/**', '*006153*polygons.json')
    glob_path = os.path.join(DATA_DIR, 'gtFine/**/**', '*polygons.json')
    tf.logging.log(tf.logging.INFO, 'Searching for pattern >%s<' % glob_path)

    json_files = tf.gfile.Glob(glob_path)

    tf.logging.log(tf.logging.INFO, 'Using %d annotation files...', len(json_files))

    random.seed(42)
    random.shuffle(json_files)
    num_examples = len(json_files)
    num_train = int(FLAGS.perc_train * num_examples)
    train_examples = json_files[:num_train]
    val_examples = json_files[num_train:]

    train_output_path = os.path.join(OUTPUT_DIR, 'cityscapes_train.record')
    val_output_path = os.path.join(OUTPUT_DIR, 'cityscapes_val.record')
    tf.logging.log(tf.logging.INFO, 'Storing (%i+%i) examples as: \n%s\n%s\n', num_train, num_examples - num_train,
                   train_output_path, val_output_path)

    create_tf_record(
        train_output_path,
        train_examples)
    create_tf_record(
        val_output_path,
        val_examples)


if __name__ == '__main__':
    tf.app.run()
