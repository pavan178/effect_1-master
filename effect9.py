from random import randrange

import cv2

# create an overlay image. You can use any image
#foreground = cv2.imread("original (1).gif")
Num = randrange(99999999999999999)
FileName = str(Num) + ".avi"
# Open the camera
cap = cv2.VideoCapture(0)



gif = cv2.VideoCapture("snow.mp4")
frame_width = int(cap.get(3))

frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(FileName, fourcc, 20.0, (frame_width, frame_height))
currentFrame = 0
img_counter = 0
print("Press Space key to take a photo")
print(FileName + "  Video recording started")
print("Press ESC key to stop recording")
# Set initial value of weights
alpha = 0.9
while cap.isOpened():
    # read the background
    ret, background = cap.read()

    ret1, foreground = gif.read()  # ret=True if it finds a frame else False.


    background = cv2.flip(background, 1)
    # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
    cv2.addWeighted(background, alpha, foreground, 1.0, 0, background)


    if ret == True:
        # Handles the mirroring of the current frame
        ##frame = cv2.flip(frame, 1)
        cv2.imshow('a', background)

        out.write(background)


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
        cv2.imwrite(img_name, background)
        print("{} written!".format(img_name))
        img_counter += 1
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

