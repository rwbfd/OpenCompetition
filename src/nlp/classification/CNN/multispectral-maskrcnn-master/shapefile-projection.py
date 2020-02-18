import shapefile
import numpy as np
import os
from PIL import Image
import cv2


def main():
    folder, track, cpt_no = 0, 'D', 1977
    os.chdir("/home/test/scene-files/Track%s" % (track))
    sf = shapefile.Reader('shapefile/d_marks_export.shp')
    shapes = sf.shapes()
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
    img = cv2.imread("img/cam0/Track_%s-Sphere-%d.jpg" % (track, cpt_no))
    with open('poses/Track_%s-Sphere-%d.pose' % (track, cpt_no)) as f:
        array = [[float(x) for x in line.split()] for line in f]
    p = np.array(array)
    p = p[:3, :]
    K, R, t_, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = np.asmatrix(-R) * np.asmatrix(t_[:3] / t_[3])
    for l in lines:
        x = np.dot(p, l)
        mask = x[2, :] > 0
        x = x[:2, mask] / x[2, mask]
        x = x.T
        if x.size == 4:
            if (x > 0).all() & (x < 2000).all():
                cv2.line(img, (int(x[0][0]), int(x[0][1])), (int(x[1][0]), int(x[1][1])), (255, 255, 255), 3)



    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        

if __name__ == '__main__':
    main()
