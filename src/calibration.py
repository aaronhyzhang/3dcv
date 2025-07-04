import numpy as np
import cv2
import glob 
import os

def calibrate_camera():
    CHESSBOARD_SIZE = (8, 6)

    imgpoints = [] # 2d
    objpoints = [] # 3d

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    path = os.path.join(os.path.dirname(__file__), '../public/chessboard')
    images = glob.glob(os.path.join(path, '*.png')) # fix paths
    count = 0

    shape = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None) # takes in gray img

        if ret == True:
            objpoints.append(objp)
            # Takes 23x23, no zero zone and terminates on eps = .001 or 30 iterations
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
            cv2.imwrite(f'../public/chessboard/img_{count}.png', img)
            count += 1

        cv2.waitKey(500)

    cv2.destroyAllWindows()

    if objpoints and imgpoints: 
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
        if ret:
            np.save('mtx.npy', mtx)
            np.save('dist.npy', dist)
            np.save('rvecs.npy', rvecs)
            np.save('tvecs.npy', tvecs)
    else:
        print("No images found")
        ret, mtx, dist, rvecs, tvecs = False, False, False, False, False

    return ret, mtx, dist, rvecs, tvecs


if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()



