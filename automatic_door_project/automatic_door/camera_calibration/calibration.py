"""
Author: Hanna Norberg
e-mail: hanna.gjelstrup.norberg@gmail.com
"""

import cv2
import numpy as np
import os

# Chessboard dimensions (internal corners)
squares_x = 8
squares_y = 8
pattern_size = (squares_x - 1, squares_y - 1)

# Prepare object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,6,0)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

SAVE_DIR = "calib_images"
os.makedirs(SAVE_DIR, exist_ok=True)
img_counter = 0

def collect_images():
    global img_counter
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'c' to capture chessboard image for calibration, ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if found:
            cv2.drawChessboardCorners(frame, pattern_size, corners, found)
            cv2.putText(frame, "Chessboard detected - Press 'c' to save", (20,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(frame, "No chessboard detected", (20,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Collect Calibration Images", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == ord('c') and found:
            img_name = os.path.join(SAVE_DIR, f"calib_{img_counter:03d}.png")
            cv2.imwrite(img_name, frame)
            print(f"Saved {img_name}")
            img_counter += 1

    cap.release()
    cv2.destroyAllWindows()

def calibrate_camera():
    images = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".png")]
    if not images:
        print("No calibration images found!")
        return None, None, None, None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            objpoints.append(objp)
            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
    
    if len(objpoints) < 5:
        print("Not enough valid calibration images (need at least 5).")
        return None, None, None, None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        print("Camera calibrated successfully!")
        print("Camera matrix:\n", mtx)
        print("Distortion coefficients:\n", dist.ravel())
        return mtx, dist, rvecs, tvecs
    else:
        print("Calibration failed.")
        return None, None, None, None

def undistort_live(mtx, dist):
    cap = cv2.VideoCapture(0)
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

def main():
    collect_images()
    mtx, dist, _, _ = calibrate_camera()
    if mtx is not None and dist is not None:
        undistort_live(mtx, dist)

if __name__ == "__main__":
    main()
