import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    listR = np.load('listR.npy')
    listT = np.load('listT.npy')
    mtx = np.load('mtx.npy')
    src_total = np.load('src_total.npy', allow_pickle=True)
    dst_total = np.load('dst_total.npy', allow_pickle=True)
    points_3d = []
    for i in range(len(listR)-1):
        Rt1 = np.concatenate((listR[i], listT[i]), axis=1)
        Rt2 = np.concatenate((listR[i+1], listT[i+1]), axis=1)

        proj1 = mtx @ Rt1
        proj2 = mtx @ Rt2

        output_3d = cv2.triangulatePoints(proj1, proj2, src_total[i], dst_total[i]) # 4d points
        output_3d = output_3d / output_3d[3]
        points_3d.append(output_3d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[0][0], points_3d[0][1], points_3d[0][2], s=1)
    plt.show()




if __name__ == "__main__":
    # main()
    all_points_3d = []
    listR = np.load('listR.npy')
    listT = np.load('listT.npy')
    mtx = np.load('mtx.npy')
    src_total = np.load('src_total.npy', allow_pickle=True)
    dst_total = np.load('dst_total.npy', allow_pickle=True)
    for i in range(len(listR)):
        # Build projection matrices for this pair
        if i == 0:
            R1, t1 = np.eye(3), np.zeros((3,1))
        else:
            R1, t1 = listR[i-1], listT[i-1]
        R2, t2 = listR[i], listT[i]
        print(R2)
        print(t2)

        P1 = mtx @ np.hstack((R1, t1))
        P2 = mtx @ np.hstack((R2, t2))

        pts1 = src_total[i].squeeze().T  # shape (2, N)
        pts2 = dst_total[i].squeeze().T  # shape (2, N)

        if pts1.shape[1] < 8:  # skip if too few points
            continue

        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        points_3d = points_4d[:3] / points_4d[3]
        all_points_3d.append(points_3d.T)  # shape (N, 3)

    # Concatenate all points
    if all_points_3d:
        all_points_3d = np.vstack(all_points_3d)
        print("Total 3D points:", all_points_3d.shape[0])
        # Plot
        mean = np.mean(all_points_3d, axis=0)
        std = np.std(all_points_3d, axis=0)
        # Keep points within 3 standard deviations of the mean
        mask = np.all(np.abs(all_points_3d - mean) < 3 * std, axis=1)
        filtered_points = all_points_3d[mask]
        centroid = np.mean(filtered_points, axis=0)
        distances = np.linalg.norm(filtered_points - centroid, axis=1)
        mask = distances < np.percentile(distances, 95)  # Keep 95% closest points
        filtered_points = filtered_points[mask]
        print("Filtered points:", filtered_points.shape[0])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(filtered_points[:,0], filtered_points[:,1], filtered_points[:,2], s=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Sparse 3D Point Cloud (Outliers Removed)')
        plt.show()
    else:
        print("No 3D points to plot!")