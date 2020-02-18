import shapefile
import numpy as np
import os, os.path
import glob
from natsort import natsorted, ns
from shapely.geometry import Polygon
import cv2
from PIL import Image


def extract_view(file_path, plane_corners_w_h, size, padding, K, R, t):



    plane_corners_c = K * np.hstack((R, t)) * plane_corners_w_h
    plane_corners_c = plane_corners_c[0:2, :] / plane_corners_c[2, :]

    dest_points = np.matrix([[padding[0], padding[0], size[0] + padding[0], size[0] + padding[0]],
                             [padding[1], size[1] + padding[1], size[1] + padding[1], padding[1]]])

    rgb_image = cv2.imread(file_path)

    H = cv2.getPerspectiveTransform(plane_corners_c.astype(np.float32).transpose(), dest_points.astype(np.float32).transpose())




    warped_img = cv2.warpPerspective(rgb_image, H, (size[0] + 2 * padding[0], size[1] + 2 * padding[1]))

    return warped_img, H

def extract_image_patches(file_path, plane_corners_w_h, img_size,  K, R, t):

    width = np.linalg.norm(plane_corners_w_h[:, 3] - plane_corners_w_h[:, 0])
    height = np.linalg.norm(plane_corners_w_h[:, 1] - plane_corners_w_h[:, 0])

    image_patches = []
    scale = 1.0
    plane_corners_c = K * np.hstack((R, t)) * plane_corners_w_h

    before_image_mask = plane_corners_c[2, :] > 0
    plane_corners_c = plane_corners_c[0:2, :] / plane_corners_c[2, :]
    within_image_mask = (plane_corners_c[0, :] >= 0) & (plane_corners_c[0, :] < img_size[0]) & \
                        (plane_corners_c[1, :] >= 0) & (plane_corners_c[1, :] < img_size[1])

    px_width = np.linalg.norm(plane_corners_c[:, 3] - plane_corners_c[:, 0])

    if np.any(before_image_mask & within_image_mask) and px_width > 0:
        px_size = (int(px_width * scale + 0.5), int(height / width * px_width * scale + 0.5))
        pad = min(int(px_size[0] * 0.2 + 0.5), int(px_size[1] * 0.2 + 0.5))
        warped_img, H = extract_view(file_path, plane_corners_w_h, px_size, (pad, pad), K, R, t)



        if np.linalg.det(H) < 0:
            warped_img = cv2.flip(warped_img, 1)



        image_patches.append(warped_img)


    return image_patches

def check_direction(corners):
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += (corners[j][0] - corners[i][0]) * (corners[j][1] + corners[i][1])
    return area/abs(area)

def main():
    folder, track = 2, 'D'

    os.chdir("/home/test/Downloads/pinhole%d" % folder)
    DIR = 'img/cam0'
    images = glob.glob(DIR+'/*.jpg')
    images = natsorted(images, alg=ns.IGNORECASE)
    sf = shapefile.Reader('shapefile/d_signs_export.shp')
    idx_fp = open('training-data/index.txt', "a+")
    signs = []
    for sh in sf.shapes():

        signs.append([])
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
                signs[-1].append(X_w_hom)

    back = [68, 69, 90, 137, 139, 140, 142, 144, 154, 155, 156, 157, 161, 165, 166, 168, 169, 170, 171, 172, 174, 175, 177, 194, 198, 201, 204, 205, 209, 228, 229,14, 79, 99, 113, 130, 133, 146, 176, 182, 184, 185, 193, 206, 207]
    for class_idx, (sign, rd) in enumerate(zip(signs, sf.records())):
        idx_fp.write('%d|%s|%s\n' % (class_idx + 1, rd[1], rd[0]))
        if rd[1]!='road_painting' and class_idx+1 not in back:


            v1 = sign[0][:3, 1] - sign[0][:3, 0]
            v2 = sign[1][:3, 0] - sign[1][:3, 1]
            normal_vector = np.cross(v1, v2)
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            y_vector = np.array([0, 0, -1])
            y_vector = y_vector - np.dot(y_vector,normal_vector.T).item() * normal_vector
            y_vector = y_vector / np.linalg.norm(y_vector)
            origin = sign[1][:, 0]

            x_vector = np.matrix(np.cross(y_vector, normal_vector))
            R = np.vstack((x_vector, y_vector, normal_vector))
            Rot = np.identity(4)
            Rot[:3, :3] = R

            corners = []
            for line in sign:
                corners.append(np.dot(Rot, line[:, 0] - origin)[:2])

            corners = np.array(corners)
            minx, miny = np.min(corners, axis=0)
            maxx, maxy = np.max(corners, axis=0)
            plane_corners = np.zeros((4, 4), dtype=np.float64)
            plane_corners[:2, 0] = np.array([minx, miny])
            plane_corners[:2, 1] = np.array([minx, maxy])
            plane_corners[:2, 2] = np.array([maxx, maxy])
            plane_corners[:2, 3] = np.array([maxx, miny])
            Rot = np.identity(4)
            Rot[:3, :3] = R.T
            origin = np.tile((origin), (1, 4)).reshape(4, 4).T
            plane_corners_w_h = np.dot(Rot, plane_corners) + origin
            print(plane_corners_w_h)
            points = []
            for cpt_no, file_path in enumerate(images):
                flag = 0
                with open('poses/Track_%s-Sphere-%d.pose' % (track, cpt_no + 1)) as f:
                    array = [[float(x) for x in line.split()] for line in f]
                p = np.array(array)
                p = p[:3, :]
                K, R, t_, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
                t = np.asmatrix(-R) * np.asmatrix(t_[:3] / t_[3])
                x = np.dot(p, plane_corners_w_h)
                mask = x[2, :] > 0
                x = x[:2, mask] / x[2, mask]
                if (x > 0).all() & (x < 2000).all():
                    x = x.T.tolist()
                    for c in x:
                        points.append(tuple(c))
                else:
                    flag = 1

                if flag == 0:
                    polygon = Polygon(points)
                    print(polygon.area)
                    if polygon.area > 1000:
                        image_patches = extract_image_patches(file_path, plane_corners_w_h, [2000, 2000], K, R, t)
                        for i in image_patches:
                            print(i.shape)
                            if i.shape[0] >= 23 and i.shape[1] >= 23 and i.shape[0] < 2000 and i.shape[1] < 2000:
                                im = Image.fromarray(i[:, :, ::-1])
                                im.save('training-data/cam0_%d_%d.png' % (cpt_no + 1, class_idx + 1))





if __name__ == '__main__':
    main()
