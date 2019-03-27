import numpy as np
import cv2
import cv2.aruco as aruco


class Marker(object):

    def __init__(self, length, center):
        self.length = length
        self.center = center
        center_edge = length / 2
        delta_x = np.array([center_edge, 0.0, 0.0])
        delta_y = np.array([0.0, center_edge, 0.0])
        corner1 = center - delta_x + delta_y
        corner2 = center + delta_x + delta_y
        corner3 = center + delta_x - delta_y
        corner4 = center - delta_x - delta_y
        self.corners = np.array([corner1, corner2, corner3, corner4])


class Goal(object):
    def __init__(self, marker_length, marker_position_relative, goal_height, goal_width, goal_depth, ball):
        self.length = marker_length
        self.marker_position = marker_position_relative  # marker position relative to center of crossbar (goalkeeper point of view)
        self.heigth = goal_height + self.marker_position[2]
        self.width = goal_width
        self.depth = goal_depth
        self.goal_area_fr = np.array([- ball.radius / 3, - self.width / 2, 0])
        self.goal_area_fl = np.array([- ball.radius / 3, self.width / 2, 0])
        self.goal_area_br = np.array([- self.depth, - self.width / 2, 0])
        self.goal_area_bl = np.array([- self.depth, self.width / 2, 0])
        self.scoring_area_fr = np.array([- ball.radius / 2, - self.width / 2, 0])
        self.scoring_area_fl = np.array([- ball.radius / 2, self.width / 2, 0])
        self.scoring_area_br = self.goal_area_br
        self.scoring_area_bl = self.goal_area_bl
        self.corners = None
        self.rvec = None
        self.tvec = None
        self.rot_mtx = None
        self.position = None
        self.goalline_position = None
        self.rotation = None
        self.rotation_z = None
        self.visible = False
        self.set = False
        self.start = None
        self.end = None
        self.goal_area_points = None
        self.scoring_area_points = None

class Field(object):

    def __init__(self, n_markers, marker_length, field_length, field_width, barrier_heigth, positions, ball, goal):
        markers_board = n_markers - 1  # n_markers includes marker for goal
        self.marker_ids = np.arange(1, n_markers)
        self.marker_list = [Marker(marker_length, positions[i]) for i in range(markers_board)]
        corner_ul = np.array([field_width, 0.0, barrier_heigth])
        corner_ur = np.array([field_width, field_length, barrier_heigth])
        corner_lr = np.array([0.0, field_length, barrier_heigth])
        corner_ll = np.array([0.0, 0.0, barrier_heigth])
        self.field_corners = np.array([[corner_ul], [corner_ur], [corner_lr], [corner_ll]])
        self.object_points = np.array([self.marker_list[i].corners for i in range(markers_board)], np.float32)
        self.ball = ball
        self.goal = goal

        self.dict = aruco.Dictionary_create(n_markers, 6)
        self.parameters = aruco.DetectorParameters_create()
        self.board = aruco.Board_create(self.object_points, self.dict, self.marker_ids)

        self.visible = False
        self.set = False
        self.ball_in_goal = False
        self.corners = None
        self.ids = None
        self.rvec = None
        self.tvec = None
        self.rot_mtx = None
        self.inv_rot_mtx = None
        self.inv_mtx = None
        self.tvec_camera = None
        self.roi_points = None
        self.radius_max = 250.0
        self.radius_min = 15.0
        self.tolerance = 0.1

    def is_visible(self, image):
        self.corners, self.ids, rejectedImgPoints = aruco.detectMarkers(image, self.dict, parameters=self.parameters)

        if self.ids is not None:
            image_draw = aruco.drawDetectedMarkers(image, self.corners, self.ids)

            if 0 in self.ids:
                self.goal.visible = True

                if len(self.ids) > 1:
                    self.visible = True
            else:
                self.visible = True
                self.goal.visible = False

        else:
            self.visible = False
            self.goal.visible = False
            image_draw = image

        return image_draw

    def set_pose(self, matrix, distortion):
        if self.visible:
            retval, self.rvec, self.tvec = aruco.estimatePoseBoard(self.corners, self.ids, self.board, matrix,
                                                                   distortion)
            self.rot_mtx, jacobian = cv2.Rodrigues(self.rvec)
            self.inv_rot_mtx = np.linalg.inv(self.rot_mtx)
            self.inv_mtx = np.linalg.inv(matrix)
            # camera coordinate in board coordinate system
            self.tvec_camera = -1 * np.matmul(self.inv_rot_mtx, self.tvec)

            self.roi_points, jacobian = cv2.projectPoints(self.field_corners, self.rvec, self.tvec, matrix, distortion)
            self.roi_points = self.roi_points.astype(int)

            distances = np.empty(0)
            for center in self.field_corners:
                distance = np.linalg.norm(center - np.reshape(self.tvec_camera, 3))
                distances = np.append(distances, [distance], axis=0)

            index_min = np.argmin(distances)
            index_max = np.argmax(distances)
            self.radius_min = radius_on_image(self.field_corners[index_max], self.ball.radius, self.tvec_camera,
                                              self.rvec, self.tvec, matrix, distortion) * (1 - self.tolerance)
            self.radius_max = radius_on_image(self.field_corners[index_min], self.ball.radius, self.tvec_camera,
                                              self.rvec, self.tvec, matrix, distortion) * (1 + self.tolerance)

            if retval != 0:
                self.set = True
            else:
                self.set = False
                self.goal.set = False
        else:
            self.set = False
            self.goal.set = False
            print('No Marker Visible')

    def set_goal(self, matrix, distortion):
        if self.set:
            if self.goal.visible:
                index = np.where(self.ids == 0)[0][0]
                self.goal.corners = [np.array(self.corners[index], np.float32)]
                self.goal.rvec, self.goal.tvec, _ = aruco.estimatePoseSingleMarkers(self.goal.corners, self.goal.length,
                                                                                    matrix, distortion)
                self.goal.rvec = self.goal.rvec.reshape(3, 1)
                self.goal.tvec = self.goal.tvec.reshape(3, 1)
                image_points_marker, jacobian = cv2.projectPoints(np.array([np.zeros((3,1))]), self.goal.rvec, self.goal.tvec, matrix, None)
                image_points_marker = image_points_marker.astype(int)
                image_point = np.array([[image_points_marker[0][0][0]], [image_points_marker[0][0][1]], [1]])
                Z_CONST = 0.000

                # calculate projection of center of goal marker on board plane
                temp_mat1 = np.matmul(self.inv_rot_mtx, np.matmul(self.inv_mtx, image_point))
                temp_mat2 = np.matmul(self.inv_rot_mtx, self.tvec)
                s = Z_CONST + temp_mat2[2]
                s /= temp_mat1[2]
                board_point = np.matmul(self.inv_rot_mtx, (s * np.matmul(self.inv_mtx, image_point) - self.tvec))

                # set z to Z_CONST again
                board_point[2] = Z_CONST

                # calculate intersection angle of line tvecCamera to boardPoint and board plane, calculate goal marker point on board plane
                board_normal = np.array([0, 0, 1])
                line_camera_point = board_point - self.tvec_camera
                alpha = np.arcsin(np.abs(np.matmul(board_normal, line_camera_point))
                                  / (np.linalg.norm(line_camera_point) * np.linalg.norm(board_normal)))

                # if clause to prevent tan(90) error
                if alpha > np.radians(89):
                    delta_board_point = 0
                else:
                    delta_board_point = (self.goal.heigth) / np.tan(alpha)

                line_camera_point_project_board = line_camera_point
                line_camera_point_project_board[2] = 0
                line_camera_point_project_board_unit = line_camera_point_project_board / np.linalg.norm(
                    line_camera_point_project_board)
                self.goal.position = board_point - delta_board_point * line_camera_point_project_board_unit

                # calculate rotation of goal in board coordinates
                self.goal.rot_mtx, jacobian = cv2.Rodrigues(self.goal.rvec)
                _, _, _, _, _, self.goal.rotation = cv2.RQDecomp3x3(np.matmul(self.goal.rot_mtx, np.transpose(self.rot_mtx)))
                rot_direction = np.matmul(self.goal.rotation, np.array([[1], [0], [0]]))
                self.goal.rotation_z = np.arctan2(rot_direction.item(1,0), rot_direction.item(0,0))
                self.goal.goalline_position = self.goal.position - self.goal.marker_position.item(0) * rot_direction
                arrow_start = self.goal.goalline_position + self.goal.length * rot_direction
                object_points = np.array([self.goal.goalline_position, arrow_start])
                image_points_arrow, jacobian = cv2.projectPoints(object_points, self.rvec, self.tvec, matrix, distortion)
                image_points_arrow = image_points_arrow.astype(int)
                self.goal.end = np.array([image_points_arrow[0][0][0], image_points_arrow[0][0][1]])
                self.goal.start = np.array([image_points_arrow[1][0][0], image_points_arrow[1][0][1]])

                # find valid goal and scoring polygon
                goal_x = self.goal.goalline_position - self.goal.position
                goal_x = goal_x / np.linalg.norm(goal_x)
                goal_y = np.array([[- goal_x[1][0]], [goal_x[0][0]], [0]])
                goalline_point = self.goal.goalline_position
                goalline_point.itemset(2, self.ball.radius)
                goal_area_fr_world = goalline_point + self.goal.goal_area_fr[0] * goal_x + self.goal.goal_area_fr[1] * goal_y
                goal_area_fl_world = goalline_point + self.goal.goal_area_fl[0] * goal_x + self.goal.goal_area_fl[1] * goal_y
                goal_area_br_world = goalline_point + self.goal.goal_area_br[0] * goal_x + self.goal.goal_area_br[1] * goal_y
                goal_area_bl_world = goalline_point + self.goal.goal_area_bl[0] * goal_x + self.goal.goal_area_bl[1] * goal_y
                scoring_area_fr_world = goalline_point + self.goal.goal_area_fr[0] * goal_x + self.goal.goal_area_fr[1] * goal_y
                scoring_area_fl_world = goalline_point + self.goal.goal_area_fl[0] * goal_x + self.goal.goal_area_fl[1] * goal_y
                scoring_area_br_world = goalline_point + self.goal.goal_area_br[0] * goal_x + self.goal.goal_area_br[1] * goal_y
                scoring_area_bl_world = goalline_point + self.goal.goal_area_bl[0] * goal_x + self.goal.goal_area_bl[1] * goal_y
                goal_area = np.array([goal_area_fr_world, goal_area_fl_world, goal_area_bl_world, goal_area_br_world])
                scoring_area = np.array([scoring_area_fr_world, scoring_area_fl_world, scoring_area_bl_world, scoring_area_br_world])
                self.goal.goal_area_points, jacobian = cv2.projectPoints(goal_area, self.rvec, self.tvec, matrix,
                                                                         distortion)
                self.goal.scoring_area_points, jacobian = cv2.projectPoints(scoring_area, self.rvec, self.tvec, matrix,
                                                                         distortion)
                self.goal.goal_area_points = self.goal.goal_area_points.astype(int)
                self.goal.scoring_area_points = self.goal.scoring_area_points.astype(int)

                self.goal.set = True
            else:
                self.goal.set = False
                print('Goal Not Visible')
        else:
            print('Field Not Set')

    def draw_pose(self, image, matrix, distortion):
        if self.set:
            image_draw = aruco.drawAxis(image, matrix, distortion, self.rvec, self.tvec, 0.1)
        else:
            image_draw = image

        return image_draw

    def draw_roi(self, image):
        if self.set:
            image_draw = cv2.polylines(image, [self.roi_points], True, (165, 0, 255))
        else:
            image_draw = image

        return image_draw

    def draw_goal(self, image):
        if self.goal.set:
            image_draw = cv2.arrowedLine(image, (self.goal.start[0], self.goal.start[1]), (self.goal.end[0], self.goal.end[1]), (240, 128, 128), 3)
            image_draw = cv2.polylines(image_draw, [self.goal.goal_area_points], True, (165, 165, 255))
        else:
            image_draw = image

        return image_draw

    def detect_scored_goal(self):
        if self.goal.set:
            if self.ball.visible:
                ball_point = self.ball.center_r
            else:

                if self.ball.kalman.found:
                    ball_point = (self.ball.kalman.state[0], self.ball.kalman.state[1])
                else:
                    ball_point = None

            if ball_point is not None:
                detect = cv2.pointPolygonTest(self.goal.scoring_area_points, ball_point, False)

                if detect > 0:
                    self.ball_in_goal = True
                else:
                    self.ball_in_goal = False


def radius_on_image(point, ball_radius, camera_position, rvec, tvec, matrix, distortion):
    center_ball = point + np.array([0.0, 0.0, ball_radius])
    line_camera_point = center_ball - np.reshape(camera_position, 3)
    random_vector = np.array([0.0, 0.0, 1.0])
    perpendicular_vector = np.cross(line_camera_point, random_vector)
    perpendicular_vector_unit = perpendicular_vector / np.linalg.norm(perpendicular_vector)
    point_on_ball = center_ball + perpendicular_vector_unit * ball_radius
    object_points = np.array([center_ball, point_on_ball])

    image_points, jacobian = cv2.projectPoints(object_points, rvec, tvec, matrix, distortion)
    image_points = image_points.astype(int)
    image_center = np.array([image_points[0][0][0], image_points[0][0][1]])
    image_on_ball = np.array([image_points[1][0][0], image_points[1][0][1]])

    radius = np.linalg.norm(image_center - image_on_ball)

    return radius