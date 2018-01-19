#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import numpy as np

#img1 = cv2.imread('../data/1010a.jpg', 0)
#declaring them globally
#sift = cv2.xfeatures2d.SURF_create(0)
#kp1, des1 = sift.detectAndCompute(img1,None)


    #order_points 
    #Formats four random cordinates passed to it for drawing a 
    #rectangle in left top, left bottom, right top, right bottom
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect

    #four_point_transform 
    #Takes image and four coordinates as input. This code 
    #blocks extract the quardilatera from an image and transform into a rectangle
    #and returns that
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped
    
    #to adjust the image size, boundary padding is done

    #SIFT_rect
    #This code block presents an algorithm for detecting a specific object based 
    #on finding point correspondences between the reference and the target 
    #image. It can detect objects despite a scale change or in-plane 
    #rotation. It is also robust to small amount of out-of-plane rotation and 
    #occlusion. SIFT_Rect takes two image, a string and a number [0-1] as a input,
    #where first image is query image (digital reference), second image is train image
    #(wild scene), string is the name of train image, and number is threshold in range 
    #[0-1]. This code block returns a warped image taken out of the scene
    #and it also save matching result as a name_result where <name> is name of
    #your image passed as a input

    #This method of object detection works best for objects that exhibit
    #non-repeating texture patterns, which give rise to unique feature matches. 
    #This technique is not likely to work well for uniformly-colored objects, 
    #or for objects containing repeating patterns. Note that this algorithm is
    #designed for detecting a specific object. For example there are two similar 
    #object in the scene, then this will not detect all the objects
def SIFT_Rect(img1, img2, qimg, thresh):
    #initialise image = img2 for warping
    image = img2

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
            if m.distance < thresh*n.distance:
                good.append(m)
    #set t from 0.40 to .70 for getting correct results

    if len(good)>5:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,[0,255,0],3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    pts = np.int32(dst)[:,0]

    warped = four_point_transform(image, pts)
    
    res = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    cv2.imwrite('{}_match_result.jpg'.format(qimg), res)
    cv2.imwrite('{}_warped.jpg'.format(qimg), warped)

    return 0