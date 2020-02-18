import shapefile
import numpy as np
import os
from PIL import Image
import cv2
import pydensecrf.densecrf as dcrf
import glob


def get_shapes(shapes):
    lines = []
    for sh in shapes:
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
                lines.append(X_w_hom)
    return lines
def indent(L, indent_right,indent_left):
    W = np.ones((3, 4), dtype=np.float64)
    W[:, 0] = (L[:, 0] + np.array([[0], [0], [indent_right]])).T
    W[:, 1] = (L[:, 1] + np.array([[0], [0], [indent_right]])).T
    W[:, 2] = (L[:, 1] + np.array([[0], [0], [-indent_left]])).T
    W[:, 3] = (L[:, 0] + np.array([[0], [0], [-indent_left]])).T
    return W

def get_intersection(p1,p2,z):

    for p in p1.tolist():
        x1, y1, z1 = p
    for p in p2.tolist():
        x2, y2, z2 = p

    if z2 < z1:
        return get_intersection(p2, p1, z)
    if (z1-z2) ==0:

        t =(z - z1) / (z2 - z1+0.0000000000000001)
    else:
        t = (z - z1)/(z2 - z1)
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return x,y


def fill_polygon(blankimage, lines, indent_right, indent_left, p):
    K, R, t_, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = np.asmatrix(-R) * np.asmatrix(t_[:3] / t_[3])

    for l in lines:
        L = np.hstack((R, t)) * l
        W = indent(L, indent_right, indent_left)
        x = np.dot(K, W)
        mask = x[2, :] > 0
        x = x[:2, mask] / x[2, mask]
        x = x.T
        if x.size == 8:
            cv2.fillPoly(blankimage, np.int32([x]), (255, 255, 255))




    return blankimage



def generate_mask(line1, line2, start=1850,stop=3724):
    for num in range(start, stop):
        print('processing %d' %(num))
        tif = cv2.imread("Track_%s-Sphere-%d-cam%d_intensity.tif" % (track, num, cam))
        img = cv2.imread("img/cam%d/Track_%s-Sphere-%d.jpg" % (cam, track, num))
        select = img
        ignore_mask = cv2.imread('ignore_masks/cam%d_ignore_mask.png' % cam)
        mask = cv2.bitwise_not(ignore_mask[:,:,0])

        with open('poses/Track_%s-Sphere-%d.pose' % (track, num)) as f:
            array = [[float(x) for x in line.split()] for line in f]
        p = np.array(array)
        p = p[cam*4:(4*cam)+3, :]


        blankimage = np.zeros((2000, 2000, 3), np.uint8)
        im1 = fill_polygon(blankimage, line1, 0.12, 0.12, p)
        im2 = fill_polygon(blankimage, line2, 0.12, 0.12, p)
        image = (im1 + im2).astype(np.uint8)

        d = dcrf.DenseCRF2D(2000, 2000, 2)
        blurred = cv2.GaussianBlur(image,(-1,-1),3)

        #Image.fromarray(blurred).show()
        px = -np.log(blurred[:, :, 0] / 255.0)
        py = -np.log(np.ones((2000, 2000), np.float32) - blurred[:, :, 0] / 255.0)
        U = np.stack((py, px), axis=2)
        U = U.transpose(2, 0, 1).reshape((2, -1))
        U = U.copy(order='C')
        d.setUnaryEnergy(U.astype(np.float32))
        d.addPairwiseGaussian(sxy=3, compat=6)
        d.addPairwiseBilateral(sxy=170, srgb=13, rgbim=select, compat=10)
        Q = d.inference(5)
        map = np.argmax(Q, axis=0).reshape((2000, 2000))
        im = map * 255
        overlay = np.stack((im, im, im), axis=2)
        final = cv2.bitwise_and(overlay, overlay, mask=mask)
        IM = Image.fromarray((blurred * 0.5 + select * 0.5).astype(np.uint8))
        im = Image.fromarray((final * 0.5 + select * 0.5).astype(np.uint8))
        msk = Image.fromarray(final.astype(np.uint8))
        msk.save('masks/cam%d_crf_%d_mask.png' % (cam, num))
        im.save('masks/cam%d_crf_%d.png' % (cam,num))
        im.show()
        #IM.save('masks/cam0_crf_%d_shp.png' % (num))





def main():
    global track, cam, total_images, n, max
    track, cam = 'D', 0
    os.chdir("/home/test/scene-files/Track%s" % (track))
    DIR = 'img/cam0'
    total_images = len(glob.glob(DIR + '/*.jpg'))
    sf = shapefile.Reader('shapefile/d_marks_export.shp')
    sr = shapefile.Reader('shapefile/d_referenceline_export.shp')


    shapes_marks = get_shapes(sf.shapes())
    shapes_reference_line = get_shapes(sr.shapes())
    generate_mask(shapes_marks ,shapes_reference_line, 1977, 1978)




if __name__ == '__main__':
    main()

