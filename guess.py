#!/usr/bin/python

import cv2
import numpy as np
from pyfann import libfann
import sys

from features import *

if __name__ == "__main__":
    ann = libfann.neural_net()
    ann.create_from_file(sys.argv[1])
    
    ori = cv2.imread(fimg, cv2.IMREAD_COLOR)

    img = ori.copy()
    img = bgr2gray(img)
    cnt = get_contours(img)
    
    f1 = get_distance_based_features(cnt)
    f2 = get_geometric_based_features(cnt)
    f3 = get_leaf_vein_features(ori.copy(), cnt)
    f4 = get_convex_hull_feature(cnt)

    f = list(f1+f2+f3).append(f4)

    print ann.run(f)
