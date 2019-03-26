import cv2
import numpy as np


def callback(value):
    pass


def setup_trackbars(thresh):
    cv2.namedWindow("Trackbars", 0)
    number = 0

    for x in thresh:

        for j in ["MIN", "MAX"]:
            counter = 0

            if number is 0:
                for k in 'HSV':
                    v = x[0][0][counter] if j == "MIN" else x[1][0][counter]
                    cv2.createTrackbar("%s_%s_%d" % (k, j, number), "Trackbars", v, 255, callback)
                    counter += 1
            else:
                k = 'H'
                v = x[0][0][counter] if j == "MIN" else x[1][0][counter]
                cv2.createTrackbar("%s_%s_%d" % (k, j, number), "Trackbars", v, 255, callback)
                counter += 1

        number += 1


def get_trackbar_values(thresh):
    all_values = []
    number = 0

    for x in thresh:
        thresh_values = []

        for j in ["MIN", "MAX"]:
            trackbar_values = []

            if number is 0:
                for k in 'HSV':
                    v = cv2.getTrackbarPos("%s_%s_%d" % (k, j, number), "Trackbars")
                    trackbar_values.append(v)
            else:
                k = 'H'
                v = cv2.getTrackbarPos("%s_%s_%d" % (k, j, number), "Trackbars")
                trackbar_values.append(v)

                for k in 'SV':
                    v = cv2.getTrackbarPos("%s_%s_%d" % (k, j, 0), "Trackbars")
                    trackbar_values.append(v)

            thresh_values.append([np.array(trackbar_values)])

        number += 1
        all_values.append(thresh_values)

    return all_values


def get_thresh(ball):
    setup_trackbars(ball.thresh)


def set_thresh(ball):
    ball.thresh = get_trackbar_values(ball.thresh)
