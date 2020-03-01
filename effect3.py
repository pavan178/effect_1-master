from math import hypot
from random import randrange

import cv2
import dlib
import numpy as np

# Loading Camera and Nose image and Creating mask

Num = randrange(99999999999999999)
FileName = str(Num) + ".avi"

cap = cv2.VideoCapture(0)
nose_image = cv2.imread("1.png")
_, frame = cap.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
frame_width = int(cap.get(3))

frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(FileName, fourcc, 20.0, (frame_width, frame_height))
currentFrame = 0
img_counter = 0
print("Press Space key to take a photo")
print(FileName + "  Video recording started")
print("Press ESC key to stop recording")
while cap.isOpened():
    _, frame = cap.read()
    ret, frame = cap.read()
    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        # Nose coordinates
        top_nose = (landmarks.part(19).x, landmarks.part(19).y)
        center_nose = (landmarks.part(27).x, landmarks.part(27).y)
        left_nose = (landmarks.part(36).x, landmarks.part(36).y)
        right_nose = (landmarks.part(45).x, landmarks.part(45).y)

        nose_width = int(hypot(left_nose[0] - right_nose[0],
                           left_nose[1] - right_nose[1]) * 1.6)
        nose_height = int(nose_width * 0.42)

        # New nose position
        top_left = (int(center_nose[0] - nose_width / 2),
                              int(center_nose[1] - nose_height / 2))
        bottom_right = (int(center_nose[0] + nose_width / 2),
                       int(center_nose[1] + nose_height / 2))


        # Adding the new nose
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(nose_area_no_nose, nose_pig)

        frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width] = final_nose
        if ret == True:
            # Handles the mirroring of the current frame
            ##frame = cv2.flip(frame, 1)
            cv2.imshow('frame', frame)

            out.write(frame)





        else:
            break

        currentFrame += 1

        key = cv2.waitKey(1)
        if key == 27:
            print("Stopped")
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            break
        elif key % 256 == 32:
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

