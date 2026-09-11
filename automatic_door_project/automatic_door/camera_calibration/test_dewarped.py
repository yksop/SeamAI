"""
Author: Hanna Norberg
e-mail: hanna.gjelstrup.norberg@gmail.com
"""

import cv2
import numpy as np

def undistort_live(mtx, dist):
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    print("Press ESC to exit undistorted live feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
        cv2.imshow("Undistorted Live Feed", undistorted)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    mtx = np.array([[307.8047385, 0, 355.19676862],
                [0, 302.44366762, 233.22849986],
                [0, 0, 1]])

    dist = np.array([-0.290496, 0.07539763, -0.00075077, -0.00159761, -0.00811828])
    undistort_live(mtx, dist)