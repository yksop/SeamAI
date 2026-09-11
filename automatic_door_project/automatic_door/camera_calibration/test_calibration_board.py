"""
Author: Hanna Norberg
e-mail: hanna.gjelstrup.norberg@gmail.com
"""

import cv2
import numpy as np

squares_x = 8
squares_y = 8
pattern_size = (squares_x - 1, squares_y - 1)

def test_chessboard(frame, pattern_size):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if found:
        print("Chessboard detected")
        cv2.drawChessboardCorners(frame, pattern_size, corners, found)
    else:
        print("Chessboard not detected. Try improving lighting or using better pattern.")
        
    cv2.imshow("Chessboard detection", frame)
    cv2.waitKey(0)
    cv2.destroyWindow('Chessboard detection')

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    print ("Camera feed started, press enter to detect chessboard, ESC to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break
        
        cv2.imshow("Camera feed", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            break
        if key == 13:
            test_chessboard(frame.copy(), pattern_size)
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    main()