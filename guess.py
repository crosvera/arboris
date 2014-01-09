#!/usr/bin/python

import cv2
import numpy as np
from pyfann import libfann
import sys

from features import *

if __name__ == "__main__":
    ann = libfann.neural_net()
    ann.create_from_file(sys.argv[1])
    
    ori = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    img = ori.copy()
    img = bgr2gray(img)
    cnt = get_contours(img)
    
    f1 = get_distance_based_features(cnt)
    f2 = get_geometric_based_features(cnt)
    f3 = get_leaf_vein_features(ori.copy(), cnt)
    f4 = get_convex_hull_feature(cnt)

    #f = list(f1+f2+f3).append(f4)
    f = f1[0], f1[1], f1[2], f1[3], f1[4], f1[5], f1[6], f1[7], f1[8], f1[9], f2[0], f2[1], f2[2], f2[3], f2[4], f3[0], f3[1], f3[2], f3[3], f3[4], f4

    print ann.run(f)
