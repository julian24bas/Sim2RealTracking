import numpy as np
import cv2


class kalman(object):

    def __init__(self):

        state_size = 5
        meas_size = 3
        contr_zize = 0

        self.filter = cv2.KalmanFilter(state_size, meas_size, contr_zize)
        self.state = np.empty([state_size, 1], np.float32)
        self.meas = np.empty([meas_size, 1], np.float32)
        self.found = False
        self.not_found = 0

        self.filter.transitionMatrix = np.asarray(self.filter.transitionMatrix)
        self.filter.measurementMatrix = np.asarray(self.filter.measurementNoiseCov)
        self.filter.processNoiseCov = np.asarray(self.filter.processNoiseCov)
        self.filter.measurementNoiseCov = np.asarray(self.filter.measurementNoiseCov)
        self.filter.errorCovPre = np.asarray(self.filter.errorCovPre)

        cv2.setIdentity(self.filter.transitionMatrix)

        self.filter.measurementMatrix = np.zeros((meas_size, state_size), np.float32)
        self.filter.measurementMatrix.itemset(0, 1.0)
        self.filter.measurementMatrix.itemset(6, 1.0)
        self.filter.measurementMatrix.itemset(14, 1.0)

        self.filter.processNoiseCov.itemset(0, 0.01)
        self.filter.processNoiseCov.itemset(6, 0.01)
        self.filter.processNoiseCov.itemset(12, 5.0)
        self.filter.processNoiseCov.itemset(18, 5.0)
        self.filter.processNoiseCov.itemset(24, 0.01)

        cv2.setIdentity(self.filter.measurementNoiseCov, 0.1)

    def estimate(self, ball, dT):
        if self.found:
            self.filter.transitionMatrix.itemset(2, dT)
            self.filter.transitionMatrix.itemset(8, dT)

            self.state = self.filter.predict()

        if not ball.visible:
            self.not_found += 1

            if self.not_found >= 100:
                self.found = False

        else:
            self.not_found = 0

            self.meas.itemset(0, ball.center_r[0])
            self.meas.itemset(1, ball.center_r[1])
            self.meas.itemset(2, ball.radius_image)

            if not self.found:
                self.filter.errorCovPre.itemset(0, 1)
                self.filter.errorCovPre.itemset(6, 1)
                self.filter.errorCovPre.itemset(12, 1)
                self.filter.errorCovPre.itemset(18, 1)
                self.filter.errorCovPre.itemset(24, 1)

                self.state.itemset(0, self.meas[0])
                self.state.itemset(1, self.meas[1])
                self.state.itemset(2, 0)
                self.state.itemset(3, 0)
                self.state.itemset(4, self.meas[2])

                self.filter.statePost = self.state

                self.found = True

            else:
                self.filter.correct(self.meas)
