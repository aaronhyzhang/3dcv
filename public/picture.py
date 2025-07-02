import cv2

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF
    cv2.imshow('frame', frame)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f'picture_{count}.png', frame)
        count += 1

cap.release()
cv2.destroyAllWindows()