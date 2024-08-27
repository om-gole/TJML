# OM GOLE, GABOR P6
#FOR IMAGE/IMAGES
import cv2
import numpy as np

# Load the images
img1 = cv2.imread('prerect1.jpg')
img2 = cv2.imread('prerect2.jpg')

# Rectify the images
H = np.eye(3)  # Replace this with your computed homography matrix
img1_rectified = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
img2_rectified = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))


# Concatenate the images
img_concatenated = np.concatenate((img1, img1_rectified, img2, img2_rectified), axis=1)

cv2.imwrite('rectify1.jpg', img1_rectified)
cv2.imwrite('rectify2.jpg', img2_rectified)
# Save the result
cv2.imwrite('concatenated.jpg', img_concatenated)

#FOR VIDEO

# Load the video
cap = cv2.VideoCapture('input.mp4')

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# Rectify each frame
H = np.eye(3)  # Replace this with your computed homography matrix
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame_rectified = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))
        out.write(frame_rectified)
    else:
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
