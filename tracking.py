import numpy as np
import cv2
from collections import deque
from kalman import kalman


class Ball(object):

    def __init__(self, diameter, color, circular_contour):
        self.radius = diameter / 2
        self.color = color
        self.contour = circular_contour
        self.thresh_rbg = np.load('threshholds_ball.npy')
        self.thresh = []
        self.kalman = kalman()

        if color is 'red':
            self.thresh.extend(([[self.thresh_rbg[0][0][0]], [self.thresh_rbg[0][0][1]]], [[self.thresh_rbg[1][0][0]], [self.thresh_rbg[1][0][1]]]))
        elif color is 'green':
            self.thresh.extend(([[[self.thresh_rbg[0][1][0]], [self.thresh_rbg[0][1][1]]]]))
        elif color is 'blue':
            self.thresh.extend(([[[self.thresh_rbg[0][2][0]], [self.thresh_rbg[0][2][1]]]]))
        else:
            print('Ball.__init__() failed: Wrong Color: Use red, green or blue')

        self.center_m = None
        self.center_r = None
        self.radius_image = None
        self.position = None
        self.visible = False
        self.moving = False
        self.last_positions = deque(maxlen=5)

    def track(self, image, field, dT):
        if field.set:
            roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(roi_mask, field.roi_points, 255)
            image = cv2.bitwise_and(image, image, mask=roi_mask)

        contours, mask = contours_color(image, self.thresh, field)
        contours = contours_size(contours, field.radius_min, field.radius_max)
        contours_matching = contours_match(contours, self.contour)

        if len(contours_matching) > 0:

            if (min(contours_matching) < 0.3):
                index_min = min(range(len(contours_matching)), key=contours_matching.__getitem__)
                c = contours[index_min]
                (self.center_r, self.radius_image) = cv2.minEnclosingCircle(c)
                #M = cv2.moments(c)
                #self.center_m = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if field.goal.set:
                    detect = cv2.pointPolygonTest(field.goal.goal_area_points, self.center_r, False)

                    if detect > 0:
                        self.visible = False
                    else:
                        self.visible = True

                else:
                    self.visible = True
            else:
                self.visible = False

        else:
            self.visible = False

        self.kalman.estimate(self, dT)

        if self.visible:
            self.last_positions.appendleft(self.center_r)

            for positions in self.last_positions:
                position_delta = np.sqrt((positions[0] - self.last_positions[0][0]) ** 2 + (positions[1] - self.last_positions[0][1]) ** 2)

                if position_delta > 2:
                    self.moving = True
                    break

            else:
                self.moving = False

        return mask

    def draw_ball(self, image):
        if self.visible:
            image_draw = cv2.circle(image, (int(self.center_r[0]), int(self.center_r[1])), int(self.radius_image),
                                    (0, 255, 255), 2)
            image_draw = cv2.circle(image_draw, (int(self.center_r[0]), int(self.center_r[1])), 5, (0, 0, 255), -1)
            # image_draw = cv2.circle(image_draw, self.center_m, 5, (0, 0, 255), -1)

        else:
            if self.kalman.found:
                image_draw = cv2.circle(image, (int(self.kalman.state[0]), int(self.kalman.state[1])), int(self.kalman.state[4]),
                                        (255, 255, 0), 2)
                image_draw = cv2.circle(image_draw, (int(self.kalman.state[0]), int(self.kalman.state[1])), 5, (255, 0, 0), -1)

            else:
                image_draw = image

        return image_draw

    def calculate_position(self, field, matrix, distortion):
        if not self.moving and self.visible and field.set:
            # pixel coordinate of ball center, z value of board
            dst = cv2.undistortPoints(np.array([[self.center_r]]), matrix, distortion, None, matrix)
            image_point = np.array([[dst[0][0][0]], [dst[0][0][1]], [1]])
            Z_CONST = 0.000

            # calculate projection of center of ball on board plane
            temp_mat1 = np.matmul(field.inv_rot_mtx, np.matmul(field.inv_mtx, image_point))
            temp_mat2 = np.matmul(field.inv_rot_mtx, field.tvec)
            s = Z_CONST + temp_mat2[2]
            s /= temp_mat1[2]
            board_point = np.matmul(field.inv_rot_mtx, (s * np.matmul(field.inv_mtx, image_point) - field.tvec))

            # set z to Z_CONST again
            board_point[2] = Z_CONST

            # calculate intersection angle of line tvecCamera to boardPoint and board plane, calculate ball point on board plane
            board_normal = np.array([0, 0, 1])
            line_camera_point = board_point - field.tvec_camera
            alpha = np.arcsin(np.abs(np.matmul(board_normal, line_camera_point))
                              / (np.linalg.norm(line_camera_point) * np.linalg.norm(board_normal)))

            # if clause to prevent tan(90) error
            if alpha > np.radians(89):
                delta_board_point = 0

            else:
                delta_board_point = (self.radius) / np.tan(alpha)

            line_camera_point_project_board = line_camera_point
            line_camera_point_project_board[2] = 0
            line_camera_point_project_board_unit = line_camera_point_project_board / np.linalg.norm(
                line_camera_point_project_board)
            self.position = board_point - delta_board_point * line_camera_point_project_board_unit
        else:
            print('Ball is moving, is occluded or field is not set.')

    def update_thresh(self):
        if self.color is 'red':
            self.thresh_rbg[0][0][0] = self.thresh[0][0][0]
            self.thresh_rbg[0][0][1] = self.thresh[0][1][0]
            self.thresh_rbg[1][0][0] = self.thresh[1][0][0]
            self.thresh_rbg[1][0][1] = self.thresh[1][1][0]
        elif self.color is 'green':
            self.thresh_rbg[0][1][0] = self.thresh[0][0][0]
            self.thresh_rbg[0][1][1] = self.thresh[0][1][0]
        elif self.color is 'blue':
            self.thresh_rbg[0][2][0] = self.thresh[0][0][0]
            self.thresh_rbg[0][2][1] = self.thresh[0][1][0]
        else:
            print('Ball.update_thresh() failed: Wrong Color: Use red, green or blue')


def contours_edge(image):
    bilateral_filtered_image = cv2.bilateralFilter(image, 5, 175, 175)
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)

    _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (len(approx) < 23) & (area > 30)):
            contour_list.append(contour)

    return contour_list


def contours_color(image, thresh, field):
    #blurred = cv2.bilateralFilter(image, 5, 175, 175)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)  #better bilateral filter?
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for x in thresh:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, x[0][0], x[1][0]))

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    mask = set_roi(mask, field)

    _, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > 10):
            contour_list.append(contour)

    return contour_list, mask


def contours_size(contours, min_radius, max_radius):
    contour_list = []
    for contour in contours:
        (contour_center, contour_radius) = cv2.minEnclosingCircle(contour)
        if ((contour_radius > min_radius) & (contour_radius < max_radius)):
            contour_list.append(contour)

    return contour_list


def contours_match(contours, matching_contour):
    contours_matching = []
    for contour in contours:
        contour_match = cv2.matchShapes(contour, matching_contour, cv2.CONTOURS_MATCH_I1, 0)
        contours_matching.append(contour_match)

    return contours_matching


def set_roi(image, field):
    image_roi = image

    if field.set:
        roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(roi_mask, field.roi_points, 255)
        image_roi = cv2.bitwise_and(image, image, mask=roi_mask)

    return image_roi