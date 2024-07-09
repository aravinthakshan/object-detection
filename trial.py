import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load images
img1 = cv.imread('myleft.jpeg', cv.IMREAD_GRAYSCALE)  # queryimage (left image)
img2 = cv.imread('myright.jpeg', cv.IMREAD_GRAYSCALE)  # trainimage (right image)

# Initialize SIFT detector
sift = cv.SIFT_create()

# Find keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Perform knn matching
matches = flann.knnMatch(des1, des2, k=2)

# Ratio test as per Lowe's paper
pts1 = []
pts2 = []
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# Convert points to numpy arrays of integers
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# Find Fundamental Matrix using RANSAC
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

# Select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Function to draw epilines and points on images
def drawlines(img1, img2, lines, pts1, pts2):
    '''Draw epilines and points on images'''
    r, c = img1.shape
    img1_color = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2_color = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0]*c)/r[1]])
        img1_color = cv.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv.circle(img1_color, tuple(pt1), 5, color, -1)
        img2_color = cv.circle(img2_color, tuple(pt2), 5, color, -1)
    
    return img1_color, img2_color

# Compute epilines
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

# Display images with epilines
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()
