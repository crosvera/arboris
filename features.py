import cv2
import numpy as np

from scipy import signal

import sys

# trackbar callback function
def trckbr_cb(x):
    pass


def bgr2gray(img):
    img[:,:,0] = img[:,:,0] * 0.1140
    img[:,:,1] = img[:,:,1] * 0.5870
    img[:,:,2] = img[:,:,2] * 0.2989
    img = img[:,:,0] + img[:,:,1] + img[:,:,2]
    return img

def get_n_vector(A, B):
    """ From two vectors return the direction vector."""
    v = (B[0] - A[0], B[1] - A[1])
    mag = np.sqrt(v[0]**2 + v[1]**2)
    v = (v[0]/mag, v[1]/mag)
    return v

def rotate_vector(v, angle=np.pi/2):
    rvx = v[0] * np.cos(angle) - v[1] * np.sin(angle)
    rvy = v[0] * np.sin(angle) + v[1] * np.cos(angle)
    return rvx, rvy

def get_centroid(contours):
    """ From a OpenCV contour, return its centroid."""
    M = cv2.moments(contours)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx,cy

def get_distance(p1, p2):
    return np.sqrt(abs(p1[0] - p2[0])**2 + abs(p1[1] - p2[1])**2)

def get_contours(gscimg, threshold=242, potion=cv2.CHAIN_APPROX_SIMPLE):
    """ From an grayscale image and a threshold value, return the contour 
        with more points"""
    ret, thldimg = cv2.threshold(gscimg, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.blur(thldimg, (3,3))
    img = cv2.Laplacian(img, cv2.CV_8U)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, potion)
    cnt = contours[0]
    for c in contours:
        if len(c) > len(cnt): cnt = c
    return cnt

def zcr(frame):
    T = len(frame) - 1
    zcr = (1 / T) * np.sum(np.signbit(np.multiply(frame[1:T], frame[0:T - 1])))
    return zcr

def get_crossing_points(dv, contour, ld=None):
    """ From a direction vector `dv' get the intersection 
        points within contour."""
    pp = []
    tp = tuple(contour[0][0])
    centroid = get_centroid(contour)
    for p in contour[1:]:
        p = tuple(p[0])
        if p[0] - tp[0] != 0:
            m = (p[1] - tp[1])/(p[0] - tp[0])
        else:
            m = (p[1] - tp[1])
        lmbd = (tp[1] - centroid[1] + m * (centroid[0] - tp[0])) / (dv[1] - m * dv[0])
        x = int(centroid[0] + lmbd * dv[0])
        y = int(centroid[1] + lmbd * dv[1])
        #check if the intersection point is in between `tp' and `p'
        if ((x <= tp[0] and x >= p[0]) or (x >= tp[0] and x <= p[0])) \
          and ((y <= tp[1] and y >= p[1]) or (y >= tp[1] and y <= p[1])):
                pp.append((x,y))
        tp = p
    pp.sort()
    return pp

def get_n_pixels(img):
    n = 0
    for i in np.nditer(img):
        if i != 0: n += 1
    return n
        
        


def get_distance_based_features(contour):
    centroid = get_centroid(contour)
    distances = np.array([get_distance(centroid, p[0]) for p in contour])[::-1]
    #get the index of the longest distance
    idx = np.argmax(distances)
    dis = np.concatenate((distances[idx:], distances[:idx]))

    fft = np.fft.fft(dis)
    mag = np.abs(fft)
    pha = np.angle(fft)

    # 10 features based on distance, fft magnitude and phase:
    # f1: average of the distance
    f1 = np.average(dis)
    # f2: standard deviation of the distance
    f2 = np.std(dis)
    # f3: zero crossing rate (zcr) of the distances from the average of the distance
    f3 = zcr(dis-f1)
    # f4: average of the fft magnitude
    f4 = np.average(mag)
    # f5: standard deviation of the fft magnitude
    f5 = np.std(mag)
    # f6: number of peaks higher than the average of the fft magnitude
    peakids = signal.find_peaks_cwt(mag, np.arange(1,10))
    npk = 0
    for p in peakids:
        if mag[p] > f4: npk += 1
    f6 = npk
    # f7: the priority of the top ten peaks of the fft magnitude
    f7 = 0 #False #TODO
    # f8: average of the fft phase
    f8 = np.average(pha)
    # f9: standard deviation of the fft phase
    f9 = np.std(pha)
    # f10: zcr of the fft phase from the average of the fft phase
    f10 = zcr(pha-f8)
    return f1,f2,f3,f4,f5,f6,f7,f8,f9,f10
    

def get_geometric_based_features(contour):
    centroid = get_centroid(contour)
    distances = np.array([get_distance(centroid, p[0]) for p in contour])
    #get the index of the longest distance
    idx = np.argmax(distances)
    ld = tuple(contour[idx][0])
    nv = get_n_vector(ld, centroid)
    #check where `nv' cross the contour
    pp = get_crossing_points(nv, contour)
    for p in pp:
        if (not ((p[0] <= ld[0] and p[0] >= centroid[0]) \
          or (p[0] >= ld[0] and p[0] <= centroid[0]))) \
            and (not ((p[1] <= ld[1] and p[1] >= centroid[1]) \
          or (p[1] >= ld[1] and p[1] <= centroid[1]))):
            break

    LL = get_distance(ld, p) # The LeafLength
    # now rotate in 90degrees the nv vector to get LeafWidth
    pv = rotate_vector(nv)
    # now check if the perpendicular vector cross the contour
    pp = get_crossing_points(pv, contour)
    tp = pp[0]
    for p in pp[1:]:
        if get_distance(tp, p) > get_distance(tp, centroid):
            break
        tp = p
    LW = get_distance(tp, p) #LeafWidth
    LA = cv2.contourArea(contour)
    LP = cv2.arcLength(contour, True)
    # 5 features based on four basic geometric features
    # f1: aspect ratio
    f1 = LL/LW
    # f2: form factor
    f2 = 4 * np.pi * LA / LP**2
    # f3: rectangularity
    f3 = LL * LW / LA
    # f4: ratio of the perimeter to eaf length
    f4 = LL / LP
    # f5: perimeter ratio of the leaf length and leaf width
    f5 = LP / (LL+LW)
    return f1,f2,f3,f4,f5
    

def get_leaf_vein_features(img, contour):
    """ from leaf image extract 5 features from the leaf main vein"""
    LA = cv2.contourArea(contour)
    img = bgr2gray(img)
    
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    opening = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k1)
    ret, opening = cv2.threshold(opening, 15, 255, cv2.THRESH_BINARY)
    #opening = bgr2gray(opening)
    LV1 = get_n_pixels(opening)

    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    opening = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k2)
    ret, opening = cv2.threshold(opening, 15, 255, cv2.THRESH_BINARY)
    #opening = bgr2gray(opening)
    LV2 = get_n_pixels(opening)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    opening = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k3)
    ret, opening = cv2.threshold(opening, 15, 255, cv2.THRESH_BINARY)
    #opening = bgr2gray(opening)
    LV3 = get_n_pixels(opening)

    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    opening = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k4)
    ret, opening = cv2.threshold(opening, 15, 255, cv2.THRESH_BINARY)
    #opening = bgr2gray(opening)
    LV4 = get_n_pixels(opening)

    return float(LV1)/LA, float(LV2)/LA, float(LV3)/LA, float(LV4)/LA, float(LV4)/LV1

    
def get_convex_hull_feature(contour):
    """ Returns the feature ratio of leaf area and convex hull area."""
    LA = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    ha = cv2.contourArea(hull)
    return (LA/ha)
    


# Test the script, from an input image return a set of 20 features
if __name__ == "__main__":
    for fimg in sys.argv[1:]:
        ori = cv2.imread(fimg, cv2.IMREAD_COLOR)

        img = ori.copy()
        img = bgr2gray(img)
        cnt = get_contours(img)
        
        f1 = get_distance_based_features(cnt)
        f2 = get_geometric_based_features(cnt)
        f3 = get_leaf_vein_features(ori.copy(), cnt)
        f4 = get_convex_hull_feature(cnt)

        #print f1+f2+f3+tuple((f4, fimg))
        print f1[0], f1[1], f1[2], f1[3], f1[4], f1[5], f1[6], f1[7], f1[8], f1[9], f2[0], f2[1], f2[2], f2[3], f2[4], f3[0], f3[1], f3[2], f3[3], f3[4], f4
        #print sys.argv[1]
