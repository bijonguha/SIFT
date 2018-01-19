#!/usr/bin/env python
"""Script to extract roi using several algorithms
"""

# Standard library
import argparse

# 3rd-party library
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import numpy as np

function_names = ['SIFT_Rect']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROI Detection Extraction')
    parser.add_argument('reference',metavar='QRY', help='Query Image')
    parser.add_argument('query', metavar='REF', help='Reference Image')
    parser.add_argument('name_of_qry_image', metavar='N_QRY', help='Name of Query Image')
    parser.add_argument('--algo', '-a', default='SIFT_Rect', metavar='ALGO',
                        choices=function_names,
                        help='Function names : ' + ' | '.join(function_names) +
                             ' (default: SIFT_Rect)')
    parser.add_argument('--thresh', '-t', default=0.70, type=float,
                        metavar='T',
                        help='Threshold between 0.45 and 0.8 (default : 0.70)')

    args = parser.parse_args()

    img1 = cv2.imread(args.reference,0)
    img2 = cv2.imread(args.query,0)

    qimg = args.name_of_qry_image

    if(args.algo == 'SIFT_Rect'):
        getattr(__import__('utils'), args.algo)(img1, img2, qimg, args.thresh)
        print(args.reference)
