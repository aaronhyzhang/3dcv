import os 
import glob 
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import calibration

def main():
    path = os.path.join(os.path.dirname(__file__), '../public/frames')
    images = sorted(glob.glob(os.path.join(path, '*.png')))
    # ret, mtx, dist, rvecs, tvecs = calibration.calibrate_camera()
    mtx = np.load('mtx.npy')
    dist = np.load('dist.npy')

    curr_R = np.eye(3)
    curr_t = np.zeros((3,1))
    listR = []
    listT = []
    src_total = []
    dst_total = []

    for i in range(8, len(images) - 1):
        img1 = cv2.imread(images[i])
        img2 = cv2.imread(images[i+1])

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)


        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        src_pts = src_pts.reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        dst_pts = dst_pts.reshape(-1, 1, 2)

        src_total.append(src_pts)
        dst_total.append(dst_pts)

        if len(matches) < 8:
            continue

        E, mask = cv2.findEssentialMat(src_pts, dst_pts, mtx, cv2.RANSAC, 0.999, 1.0)
        _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, mtx)

        curr_R = R @ curr_R
        curr_t = R @ curr_t + t

        listR.append(curr_R)
        listT.append(curr_t)

        np.save('listR.npy', listR)
        np.save('listT.npy', listT)
        np.save('src_total.npy', np.array(src_total, dtype=object), allow_pickle=True)
        np.save('dst_total.npy', np.array(dst_total, dtype=object), allow_pickle=True)


        # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # cv2.imwrite(f'../public/matches/matches_{i}.png', img3)
        # cv2.imshow('img3', img3)



# GPT code to plot 2d and 3d maps:
def plot(listR, listT):
    if listR and listT:
        # Extract camera positions
        positions = np.array([t.flatten() for t in listT])
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot camera path
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-o', linewidth=2, markersize=4)
        
        # Mark start and end points
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, label='End')
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title('Camera Trajectory in 3D Space')
        ax.legend()
        
        # Make axes equal
        ax.set_box_aspect([1,1,1])
        
        plt.show()
        
        print(f"Processed {len(listT)} camera poses")
        print(f"Camera moved from {positions[0]} to {positions[-1]}")

        # Create 2D projections
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # XY view (top-down)
        ax1.plot(positions[:, 0], positions[:, 1], 'b-o')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Top View (XY)')
        ax1.grid(True)

        # XZ view (side view)
        ax2.plot(positions[:, 0], positions[:, 2], 'r-o')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('Side View (XZ)')
        ax2.grid(True)

        # YZ view (front view)
        ax3.plot(positions[:, 1], positions[:, 2], 'g-o')
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        ax3.set_title('Front View (YZ)')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()