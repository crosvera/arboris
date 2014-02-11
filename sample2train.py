import sys
import os

from sklearn import svm
from sklearn import neighbors
from sklearn import decomposition
import numpy as np
import cv2

from features import *


def sample2train():
    folder = "sandbox"
    samples = [{"file":"00.sample", "target":0}, \
               {"file":"01.sample", "target":1}, \
               {"file":"02.sample", "target":2}, \
               {"file":"03.sample", "target":3}, \
               {"file":"04.sample", "target":4}, \
               {"file":"05.sample", "target":5}, \
               {"file":"06.sample", "target":6}, \
               {"file":"07.sample", "target":7}, \
               {"file":"08.sample", "target":8}, \
               {"file":"09.sample", "target":9}, \
               {"file":"10.sample", "target":10}, \
               {"file":"11.sample", "target":11}, \
               {"file":"12.sample", "target":12}, \
               {"file":"13.sample", "target":13}, \
               {"file":"14.sample", "target":14}, \
               {"file":"15.sample", "target":15}, \
               {"file":"16.sample", "target":16}, \
               {"file":"17.sample", "target":17}, \
               {"file":"18.sample", "target":18}, \
               {"file":"19.sample", "target":19}]

    data = []
    target = []
    for s in samples:
        f = open(os.path.join(folder, s['file']))
        for l in f.readlines():
            data.append([np.float(n) for n in l.strip().split()])
            target.append(int(s['target']))
        f.close()

    return np.array(data), np.array(target)
    #return data, target

#def sample2train(argv):
#    print "Retrieving data from samples"
#    data = []
#    target = []
#    l = len(argv)
#    for i in range(l):
#        f = open(argv[i])
#        for l in f.readlines():
#            data.append(l.strip().split())
#            target.append(i)
#
#    return np.array(data), np.array(target)


def train_network(data, target):
    #print "Training started"
    #data, target = sample2train()
    #clf = svm.SVC()

    # dimension reduction
    pca = decomposition.RandomizedPCA(n_components=21, whiten=True)
    pca.fit(data)
    data_pca = pca.transform(data)

    clf = svm.SVC(kernel="rbf", C=1000000.0, gamma=0.00000001)
    #clf = svm.NuSVC(kernel="rbf")
    #clf = svm.SVR(C=1.0, epsilon=0.2)
    #clf = svm.LinearSVC(multi_class='crammer_singer')
    #clf = svm.NuSVC(kernel='poly', degree=4)
    clf.fit(data_pca, target)
    #print "Training ended"
    return clf, pca



def set_knn():
    data, target = sample2train()
    knn = neighbors.KNeighborsClassifier()
    knn.fit(data, target)

    return knn

if __name__ == "__main__":
    data, target = sample2train()

    clf, pca = train_network(data, target)
    #clf = set_knn()
   
    #print "guessing..."
    ori = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    img = ori.copy()
    img = bgr2gray(img)
    cnt = get_contours(img)

    f1 = get_distance_based_features(cnt)
    f2 = get_geometric_based_features(cnt)
    f3 = get_leaf_vein_features(ori.copy(), cnt)
    f4 = get_convex_hull_feature(cnt)

    f = f1[0], f1[1], f1[2], f1[3], f1[4], f1[5], f1[6], f1[7], f1[8], f1[9], f2[0], f2[1], f2[2], f2[3], f2[4], f3[0], f3[1], f3[2], f3[3], f3[4], f4
    f = list(f)

    test_pca = pca.transform(f)

    print sys.argv[1].split("/")[-1], clf.predict(test_pca)


