import sys

import cv2
import numpy
import matplotlib.pyplot as plt

from features import *


def arboris_wait_exit():
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


def test_geometric_features(argv):
    for fimg in argv[1:]:
        ori = cv2.imread(fimg, cv2.IMREAD_COLOR)

        img = ori.copy()
        gsc = ori.copy()
        gsc = bgr2gray(gsc)
            
        print fimg

        #cnt = get_contours(gsc, cv2.CHAIN_APPROX_NONE)
        cnt = get_contours(gsc)
        cv2.drawContours(img, [cnt], 0, (255,0,0), 2)

        centroid = get_centroid(cnt)
        print "centroid:", centroid
        cv2.circle(img, centroid, 4, (0,0,255), -1)

        
        distances = np.array([get_distance(centroid, p[0]) for p in cnt])
        idx = np.argmax(distances)
        ld = tuple(cnt[idx][0])
        print "LD:", ld
        cv2.circle(img, ld, 4, (0,125,255), -1)
        nv = get_n_vector(ld, centroid)
        pp = get_crossing_points(nv, cnt)
        for p in pp:
            if (not ((p[0] <= ld[0] and p[0] >= centroid[0]) \
              or (p[0] >= ld[0] and p[0] <= centroid[0]))) \
                and (not ((p[1] <= ld[1] and p[1] >= centroid[1]) \
              or (p[1] >= ld[1] and p[1] <= centroid[1]))):
                break

        cv2.circle(img, p, 4, (50,50,255), -1)
        cv2.line(img, ld, p, (0,255,100), 3)



        pv = rotate_vector(nv)
        pp = get_crossing_points(pv, cnt)
        tp = pp[0]
        for p in pp[1:]:
            if get_distance(tp, p) > get_distance(tp, centroid):
                break
            tp = p
        cv2.line(img, tp, p, (0,255,255), 3)
        print "pp", pp
        print "p, tp", p, tp

        
        
        cv2.imshow("Arboris Test", img)

        arboris_wait_exit()


def test_distance_based_features(argv):
    for fimg in argv[1:]:
        ori = cv2.imread(fimg, cv2.IMREAD_COLOR)

        img = ori.copy()
        gsc = ori.copy()
        gsc = bgr2gray(gsc)
        cnt = get_contours(gsc)
        centroid = get_centroid(cnt)
        print "centroid:", centroid
        distances = np.array([get_distance(centroid, p[0]) for p in cnt])
        idx = np.argmax(distances)
        dis = np.concatenate((distances[idx:], distances[:idx]))

        fft = np.fft.fft(dis)
        mag = np.abs(fft)
        pha = np.angle(fft)

        plt.plot(pha)
        plt.ylabel("PHA FFT")
        plt.show()


def test_vein_based_features(argv):
    for fimg in argv[1:]:
        cv2.namedWindow("Veins")
        #cv2.createTrackbar("Threshold", "Veins", 0, 255, trckbr_cb)
        cv2.createTrackbar("Kernel", "Veins", 1, 8, trckbr_cb)
        ori = cv2.imread(fimg, cv2.IMREAD_COLOR)

        while(1):
            img = ori.copy()
            gsc = ori.copy()
            gsc = bgr2gray(gsc)
            cnt = get_contours(gsc)

            kr = cv2.getTrackbarPos("Kernel", "Veins")
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kr,kr))
            opening = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)
            #thld = cv2.getTrackbarPos("Threshold", "Veins")
            ret, img = cv2.threshold(opening, 15, 255, cv2.THRESH_BINARY)
            #opening = cv2.Laplacian(img, cv2.CV_8U)

            #cv2.imshow("Veins", opening)
            img = bgr2gray(img)
            cv2.imshow("Veins", img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break



if __name__ == "__main__":
    #test_geometric_features(sys.argv)
    #test_distance_based_features(sys.argv)
    test_vein_based_features(sys.argv)
