import cv2
import os
import calibration


def take_video():
    cap = cv2.VideoCapture(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        writer.write(frame)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            print("Recording...")
            writer.write(frame)
        elif key == ord('q'):
            break

    writer.release()
    cap.release()

    cv2.destroyAllWindows()

def undistort_video(mtx, dist):
    output_path = '../public/frames'
    video = cv2.VideoCapture('output.mp4')
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{total_frames} frames")
    frame_idx = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            curr_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
            if curr_frame % 10 == 0:
                frame_path = os.path.join(output_path, f'frame_{frame_idx}.png')
                h, w = frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
                x, y, w, h = roi
                undistorted_frame = undistorted_frame[y:y+h, x:x+w]
                cv2.imwrite(frame_path, undistorted_frame)
                frame_idx += 1
        else:
            break
    video.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # take_video()
    ret, mtx, dist, rvecs, tvecs = calibration.calibrate_camera()
    if ret:
        undistort_video(mtx, dist)
    else: 
        print("Calibration failed")