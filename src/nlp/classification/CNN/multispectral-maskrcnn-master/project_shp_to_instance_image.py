from glob import glob
import os
import shapefile
import numpy as np
import cv2
from PIL import Image, ImageDraw
from helpers import get_projection

track = 'D'
INPUT_FOLDER_IM = '/media/test/Data/alplab/Track%s' % track
TEMPLATE_IM = 'Track_%s-Sphere-{}-{}.jpg' % track
TEMPLATE_IMG = 'Track_%s-Sphere-{}.jpg' % track
TEMPLATE_CHANNEL = 'Track_%s-Sphere-{}-{}_{}.tif' % track

INPUT_FOLDER_POSE = '/media/test/Data/alplab/Track%s/poses' % track
TEMPLATE_POSE = 'Track_%s-Sphere-{}.pose' % track

OUTPUT_FOLDER = '/media/test/Data/alplab/Track%s/instancesNew' % track
TEMPLATE_OUTPUT = 'Track_%s-Sphere-{}-{}_IIMG.png' % track
TEMPLATE_DBG_OUTPUT = 'Track_%s-Sphere-{}_DBG.png' % track

##########################################################################################
# DO _N O T_ USE CAM2 or CAM3 with barriers between directions (e.g. Track-D 2000-2300)####
##########################################################################################
USED_CAMERAS = ['cam0', 'cam1']
#USED_CAMERAS = ['cam2', 'cam3']
#USED_CAMERAS = ['cam0', 'cam1', 'cam2', 'cam3']

# define a maximal deviation from the camera center (left and right [for oncoming traffic + traffic signs], max distance- currently same for all)
CAMERA_MIN_MAX = {'cam0': (-10, 500, 100), 'cam1': (-500, 2, 100), 'cam2': (10, 500, 100), 'cam3': (-500, 0, 100)}

# depreciated as records hold informations abot the lane markings
# define a maximum factor w*d/h of a shape (get rid of lane markings)
# MAX_DEPTH_RATIO = 2

# entries within the pose file for each camera
POSE_ENTRIES = {'cam0': [0, 16], 'cam1': [16, 32], 'cam2': [32, 48], 'cam3': [48, 64]}

MIN_MASK_PX = 500

# get ids of all available image files
im_files = glob(os.path.join(INPUT_FOLDER_IM, 'img/cam0', '*.jpg'))
ids = [int(t.split('-')[-1].split('.')[0]) for t in im_files]
#ids = [int(parse(TEMPLATE_IM, t.split('/')[-1]).fixed[0]) for t in im_files]
ids.sort()

# select part without construction site (because track-d is only partially covered there, many traffic signs not annotated)
ids = ids[1972:2300]  # for cam0+1 in track D
#ids = ids[2300:] # for all cams in track D
# ids = ids[2000:3690]
# ids = ids[3155:3170]
# ids = (ids[3164], )
# ids = (ids[3735], )
# ids = ids[3690:]

sf = shapefile.Reader('/media/test/Data/alplab/Track%s/shapefile/s_signs.shp' % track)
shapes = sf.shapes()
records = sf.records()

# %%

# create output folder for each camera
for cam in USED_CAMERAS:
    try:
        os.makedirs(os.path.join(OUTPUT_FOLDER, cam))
    except OSError:
        pass

INST_BASE = 20000

for id in ids:

    for cam in USED_CAMERAS:
        ignore_mask = cv2.imread('/media/test/Data/alplab/Track%s/ignore_masks/%s_ignore_mask.png' % (track, cam))
        mask = cv2.bitwise_not(ignore_mask[:, :, 0])
        # load current image
        intensity = Image.open(os.path.join(INPUT_FOLDER_IM, 'intensity', cam, TEMPLATE_CHANNEL.format(id, cam,'intensity')))
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

        grayscale = Image.open(os.path.join(INPUT_FOLDER_IM, 'img', cam, TEMPLATE_IMG.format(id))).convert('L')
        grayscale = np.array(grayscale)

        im = np.dstack((grayscale, depth, intensity))
        im = im.astype('uint8')
        size = im[:, :, 0].shape
        Image.fromarray(im).save(os.path.join('/media/test/Data/alplab/Track%s/GDI_img' % track, cam, TEMPLATE_IM.format(id, cam)))
        #Image.fromarray(im).show()
        #Image.fromarray(im).save('/media/test/Data/color1.png')





        # copy image for drawing line markings
        instanceColor = INST_BASE

        dbg_im = im.copy()
        dbg_im = Image.fromarray(dbg_im)
        ddraw = ImageDraw.Draw(dbg_im)

        draw_im = Image.new("I", size, 0)
        draw = ImageDraw.Draw(draw_im)

        pose_file = os.path.join(INPUT_FOLDER_POSE, TEMPLATE_POSE.format(id))
        K, R, t, _ = get_projection(pose_file, POSE_ENTRIES[cam])
        T_w_c = np.hstack((R, t))

        (xmin, xmax, max_dist) = CAMERA_MIN_MAX[cam]

        has_polys = False
        active_polys = []
        for sh, rc in zip(shapes, records):
            # skip road paintings
            if rc[1] == 'road_painting':
                continue

            # check if current line consists of multiple parts
            '''
            Parts simply group collections of points into shapes. If the shape
            record has multiple parts this attribute contains the index of the first
            point of each part. If there is only one part then a list containing 0 is
            returned.
            '''

            shape_visible = False
            shape_partialy_invalid = False
            shape_poly = []
            shape_d = []
            shape_c = []

            for i in range(len(sh.parts)):
                # get start & end point index for current part
                start_idx = sh.parts[i]

                if (i + 1) < len(sh.parts):
                    # end_idx is start_point index of the next part minus 1
                    end_idx = sh.parts[i + 1] - 1
                else:
                    # if this is the last part then end_idx = last point
                    end_idx = len(sh.points) - 1

                # iterate over all line segments within one part
                for j in range(start_idx, end_idx):
                    # get start and end point coordinates for current line segment

                    # shape for lane markings is of type polylinez, which means for each
                    # (x,y) point, a z-coordinate is available
                    start_p = np.array([sh.points[j][0], sh.points[j][1], sh.z[j]])
                    end_p = np.array([sh.points[j + 1][0], sh.points[j + 1][1], sh.z[j + 1]])

                    # create matrix containing homogenous coordinates for start & end point
                    X_w_hom = np.ones((4, 2), dtype=np.float64)
                    X_w_hom[:3, 0] = start_p
                    X_w_hom[:3, 1] = end_p

                    # project into camera coordinates
                    X_c = np.asarray(T_w_c * np.asmatrix(X_w_hom))

                    # for the moment we keep only lines if start and
                    # end point are in front of the camera
                    if X_c[2, 0] > 0 and X_c[2, 1] > 0:
                        # and if distance is smaller then MAX_DIST
                        dist = np.linalg.norm(X_c, axis=0)
                        if dist[0] < max_dist and dist[1] < max_dist:
                            # convert to pixel coordinates
                            x_im_hom = np.asarray(K * np.asmatrix(X_c))
                            x_im = (np.around(x_im_hom[0:2, :] / x_im_hom[2, :])).astype(np.int32)

                            # check if start and end point are visible in the image
                            inlier_flags = \
                                (x_im[0, :] >= 0) & (x_im[0, :] < im.shape[1]) & \
                                (x_im[1, :] >= 0) & (x_im[1, :] < im.shape[0])

                            if np.sum(inlier_flags) >= 1:
                                shape_visible = True

                                line = [x_im[0, 0], x_im[1, 0], x_im[0, 1], x_im[1, 1]]
                                [shape_poly.append(i) for i in zip(line[::2], line[1::2])]
                                shape_d.append([dist[0], dist[1]])
                                shape_c.append(X_c)

                                ddraw.line(line, fill=(255, 0, 0), width=3)
                                # draw.line(line, fill=(instanceColor), width=3)

                                # now exclude if on left hand side (oncoming traffic)
                                if X_c[0, 0] < xmin or X_c[0, 1] < xmin \
                                        or X_c[0, 0] > xmax or X_c[0, 1] > xmax:
                                    # ddraw.line(line, fill=(255,255,0), width=3)
                                    shape_partialy_invalid = True

            if shape_visible:
                has_polys = True

                s_np = np.array(shape_poly)
                # aerea = cv2.contourArea(s_np, True) currently aereas are/seem not oriented..
                aerea = cv2.contourArea(s_np)

                shape_dist_avg = np.average(shape_d)

                #                #working but due to records in shapefile useless:
                #                shape_c_np = np.hstack(shape_c)
                #                shape_c_dim = np.max(shape_c_np, axis=1) - np.min(shape_c_np, axis=1)
                #
                #                if cam in ['cam0', 'cam2']:
                #                  depth_ratio = shape_c_dim[2]/(shape_c_dim[0]*shape_c_dim[1])
                #                elif cam in ['cam1', 'cam3']:
                #                  depth_ratio = shape_c_dim[0]/(shape_c_dim[2]*shape_c_dim[1])

                color = (255, 255, 0)

                if shape_partialy_invalid:
                    color = (255, 0, 0)
                #                elif depth_ratio > MAX_DEPTH_RATIO:
                #                  color = (255,0,255)
                elif aerea > MIN_MASK_PX:
                    print(cam, 'ID:', id)
                    color = (0, 255, 00)

                    active_polys.append((shape_dist_avg, shape_poly, instanceColor))
                    instanceColor += 1

                ddraw.polygon(shape_poly, fill=color)

        if has_polys:
            dbg_im.save(os.path.join(OUTPUT_FOLDER, cam, TEMPLATE_DBG_OUTPUT.format(id)))

        if len(active_polys) > 0:
            for s in sorted(active_polys, key=lambda x: -x[0]):
                draw.polygon(s[1], fill=s[2])

            draw_im = cv2.bitwise_and(np.array(draw_im), np.array(draw_im), mask=mask)
            draw_im = Image.fromarray(draw_im)
            draw_im.save(os.path.join(OUTPUT_FOLDER, cam, TEMPLATE_OUTPUT.format(id, cam)))

    del draw
