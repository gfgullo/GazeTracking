from camera import Camera
import cv2

camera = Camera()

while True:

    img = camera.capture(True, show_rects=True)
    cv2.imshow("Gaze Tracking", img)

    if cv2.waitKey(1) == ord("q"):
        break