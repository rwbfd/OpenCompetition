from glob import glob
import os
import shapefile
import numpy as np
import cv2
from PIL import Image, ImageDraw
from helpers import get_projection
from libtiff import TIFF

track = 'E'
INPUT_FOLDER_IM = '/media/test/Data/alplab/Track%s' % track
TEMPLATE_IM = 'Track_%s-Sphere-{}-{}.jpg' % track
TEMPLATE_OUT = 'Track_%s-Sphere-{}-{}.tif' % track
TEMPLATE_IMG = 'Track_%s-Sphere-{}.jpg' % track
TEMPLATE_CHANNEL = 'Track_%s-Sphere-{}-{}_{}.tif' % track

INPUT_FOLDER_POSE = '/media/test/Data/alplab/Track%s/poses' % track
TEMPLATE_POSE = 'Track_%s-Sphere-{}.pose' % track

OUTPUT_FOLDER = '/media/test/Data/alplab/Track%s/instancesNew' % track
TEMPLATE_OUTPUT = 'Track_%s-Sphere-{}-{}_IIMG.png' % track
TEMPLATE_DBG_OUTPUT = 'Track_%s-Sphere-{}_DBG.png' % track


USED_CAMERAS = ['cam0', 'cam1']
#USED_CAMERAS = ['cam2', 'cam3']
#USED_CAMERAS = ['cam0', 'cam1', 'cam2', 'cam3']
CAMERA_MIN_MAX = {'cam0': (-10, 500, 100), 'cam1': (-500, 2, 100), 'cam2': (10, 500, 100), 'cam3': (-500, 0, 100)}

POSE_ENTRIES = {'cam0': [0, 16], 'cam1': [16, 32], 'cam2': [32, 48], 'cam3': [48, 64]}

MIN_MASK_PX = 500

# get ids of all available image files
im_files = glob(os.path.join(INPUT_FOLDER_IM, 'img/cam0', '*.jpg'))
ids = [int(t.split('-')[-2]) for t in im_files]
#ids = [int(parse(TEMPLATE_IM, t.split('/')[-1]).fixed[0]) for t in im_files]
ids.sort()

ids = ids[1972:2300]  # for cam0+1 in track D
#ids = ids[2300:] # for all cams in track D

sf = shapefile.Reader('/media/test/Data/alplab/Track%s/shapefile/s_signs.shp' % track)
shapes = sf.shapes()
records = sf.records()
for cam in USED_CAMERAS:
    try:
        os.makedirs(os.path.join(OUTPUT_FOLDER, cam))
    except OSError:
        pass

INST_BASE = 20000

for id in ids:
    print(id)
    for cam in USED_CAMERAS:
        ignore_mask = cv2.imread('/media/test/Data/alplab/Track%s/ignore_masks/%s_ignore_mask.png' % (track, cam))
        mask = cv2.bitwise_not(ignore_mask[:, :, 0])
        intensity = Image.open(
            os.path.join(INPUT_FOLDER_IM, 'intensity', cam, TEMPLATE_CHANNEL.format(id, cam, 'intensity')))
        intensity = np.array(intensity)
        max = np.amax(intensity)
        min = np.amin(intensity)
        intensity = np.subtract(intensity, min)
        intensity = intensity / (max - min)
        intensity = intensity * 255

        depth = Image.open(os.path.join(INPUT_FOLDER_IM, 'depth', cam, TEMPLATE_CHANNEL.format(id, cam, 'depth')))
        depth = np.array(depth)
        max = np.amax(depth)
        min = np.amin(depth)
        depth = np.subtract(depth, min)
        depth = depth / (max - min)
        depth = depth * 255

        rgb = Image.open(os.path.join(INPUT_FOLDER_IM, 'img', cam, TEMPLATE_IM.format(id,cam)))
        rgb = np.array(rgb)

        im = np.dstack((rgb, depth, intensity))
        im = im.astype('uint8')
        size = im[:, :, 0].shape
        tif = TIFF.open(os.path.join('/media/test/Data/alplab/Track%s/RGBDI_img' % track, cam, TEMPLATE_OUT.format(id, cam)), mode='w')
        tif.write_image(im)


