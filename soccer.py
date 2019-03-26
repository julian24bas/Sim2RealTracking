import pyrealsense2 as rs
import numpy as np
import cv2

from matchfield import Field
from matchfield import Goal
from tracking import Ball
import threshhold

# set marker, board and ball properties
n_markers = 5  # number of markers in dictionary
marker_length = 0.07  # edge length of single marker in meter
marker_separation_x = 1.28  # distance between two markers in x direction in meter (0.13572) (1.542)
marker_separation_y = 0.5995  # distance between two markers in y direction in meter (0.216577) (0.772)
marker_heigth = 0.075  # distance of marker to ground
field_length = 0.645
field_width = 1.2
barrier_heigth = 0.051
ball_diameter = 0.057  # in meter
goal_height = 0.11  # z of marker on goal
goal_depth = 0.073
goal_width = 0.184  # inner post distance
goal_marker_position = np.array([- marker_length / 2 - 0.005, 0.0, 0.0])  # goalkeeper point of view: x pointing forward, z zero level on goal heigth
img_contour = cv2.imread('shape_matching_circle.png', 0)
ret, img_contour = cv2.threshold(img_contour, 127, 255, 0)
_, img_contours, _ = cv2.findContours(img_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ball_contour = img_contours[0]

marker_center4 = np.array([- (0.005 + (marker_length / 2)), 0.022, marker_heigth])
marker_center1 = marker_center4 + np.array([0.0, marker_separation_y, 0])
marker_center2 = marker_center4 + np.array([marker_separation_x, marker_separation_y, 0])
marker_center3 = marker_center4 + np.array([marker_separation_x, 0.0, 0])

marker_centers = np.array([[marker_center1], [marker_center2], [marker_center3], [marker_center4]])

ball = Ball(ball_diameter, 'red', ball_contour)
goal = Goal(marker_length, goal_marker_position, goal_height, goal_width, goal_depth, ball)
field = Field(n_markers, marker_length, field_length, field_width, barrier_heigth, marker_centers, ball, goal)
hsv = False
scored = False
current_time = 0.0
wait_timer = 0.0
score = 0
score_string = '0:0'

# loading calibration matrix
calibrationData = np.load('calibration.npy')
mtx = calibrationData[0]
dist = calibrationData[1]

#save video to file
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1920,1080))

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
##config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        colorFrame = frames.get_color_frame()
        if not colorFrame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(colorFrame.get_data())
        ##color_image_undist = cv2.undistort(color_image, mtx, dist, None, mtx) #if activated set color_image to this and dist to None
        color_image_draw = color_image
        prior_time = current_time
        current_time = cv2.getTickCount();
        delta_time = (current_time - prior_time) / cv2.getTickFrequency();

        mask = ball.track(color_image, field, delta_time)
        ball.calculate_position(field, mtx, dist)
        color_image_draw = field.is_visible(color_image_draw)
        color_image_draw = field.draw_roi(color_image_draw)
        color_image_draw = field.draw_goal(color_image_draw)
        color_image_draw = field.draw_pose(color_image_draw, mtx, dist)
        color_image_draw = ball.draw_ball(color_image_draw)
        cv2.putText(color_image_draw, score_string, (10, 110), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)

        field.detect_scored_goal()

        # crop color frame and hsv threshhold and create one image
        ratio = 1920.0 / color_image_draw.shape[1]
        dim = (1920, int(color_image_draw.shape[0] * ratio))
        outputColor = cv2.resize(color_image_draw, dim, interpolation=cv2.INTER_AREA)
        outputThresh = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
        outputThresh = cv2.cvtColor(outputThresh, cv2.COLOR_GRAY2BGR)
        outputStacked = np.hstack((outputColor, outputThresh))

        #save frame to file
        #out.write(outputColor)

        # Display the resulting frame
        cv2.imshow('frame', outputStacked)
        k = cv2.waitKey(1)

        if k == ord('s'):
            field.set_pose(mtx, dist)

        if k == ord('g'):
            field.set_goal(mtx, dist)

        if k == ord('h'):  # wait for 'h' key to set hsv values

            if hsv:
                hsv = False
                cv2.destroyWindow("Trackbars")

            else:
                threshhold.get_thresh(ball)
                hsv = True

        if hsv:
            threshhold.set_thresh(ball)

        if field.ball_in_goal:

            if not scored:
                scored = True
                score += 1
                score_string = '0:%d' % score

        else:

            if scored:
                wait_timer += delta_time

                if wait_timer > 0.5:
                    wait_timer = 0.0
                    scored = False

        if k == ord('q'):
            ball.update_thresh()
            np.save('threshholds_ball', ball.thresh_rbg)
            break


finally:

    # Stop streaming
    #out.release()
    pipeline.stop()

