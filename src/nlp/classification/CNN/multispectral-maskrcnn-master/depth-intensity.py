import os
import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from PIL import Image
def main():
    cam, track, cpt_no = 0, 'D', 316
    os.chdir("/home/test/Downloads/depth_intensity/cam%d" % cam)
    #background = cv2.imread("Track_%s-Sphere-%d-cam%d_intensity.tif" % (track,cpt_no,cam))
    background = cv2.imread("Track_%s-Sphere-%d.jpg" % (track, cpt_no))
    #cv2.line(background, (1310, 1380), (1240, 1280), (255, 255, 255), 15)
    d = dcrf.DenseCRF2D(2000, 2000, 2)
    lineimg = np.zeros((2000, 2000, 3), np.uint8)
    cv2.line(lineimg, (1310, 1380), (1240, 1280), (255, 255, 255), 15)
    lineimg = cv2.GaussianBlur(lineimg, (5, 5), 0)
    px = -np.log(lineimg[:,:,0]/255.0)
    py = -np.log(np.ones((2000,2000), np.float32) - lineimg[:,:,0]/255.0)
    U = np.stack((py,px), axis=2)
    U = U.transpose(2, 0, 1).reshape((2, -1))
    U = U.copy(order='C')
    d.setUnaryEnergy(U.astype(np.float32))
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=background, compat=10)
    Q = d.inference(5)
    map = np.argmax(Q, axis=0).reshape((2000, 2000))
    im = map * 255
    overlay = np.stack((im,im,im), axis=2)
    im = Image.fromarray((overlay * 0.5 + background * 0.5).astype(np.uint8))
    im.show()


















    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image', background)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
