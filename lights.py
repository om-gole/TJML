import cv2
import numpy as np

# Access the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
cap = cv2.imread("lights5.png")
# Define ranges of colors for Christmas lights in HSV
colors = {
    'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
    'green': (np.array([35, 50, 50]), np.array([90, 255, 255])),  # Adjusted green range
    'blue': (np.array([100, 100, 100]), np.array([130, 255, 255])),  # Adjusted blue range
}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define brightness parameter
    brightness = -150

    # Modify the value channel to change brightness
    hsv[:,:,2] = [[max(pixel - brightness, 0) if pixel < 190 else min(pixel + brightness, 255) for pixel in row] for row in hsv[:,:,2]]

    # Convert back to BGR
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the frame to HSV for better color thresholding
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Initialize an empty mask to collect all identified colors
    mask = np.zeros_like(frame[:, :, 0])

    # Detect colors of Christmas lights
    light_positions = []  # Store positions of lights

    for color_name, (lower, upper) in colors.items():
        # Threshold the HSV image to get only specified color ranges
        color_mask = cv2.inRange(hsv, lower, upper)

        # Apply morphological operations to enhance the mask for each color
        kernel = np.ones((3, 3), np.uint8)  # Smaller kernel for fine adjustment
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        # Combine the masks for different colors
        mask = cv2.bitwise_or(mask, color_mask)

    # Apply threshold to identify bright areas
    _, threshold = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)

    # Perform morphological operations (erosion and dilation) to isolate individual lights
    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.erode(threshold, kernel, iterations=1)
    threshold = cv2.dilate(threshold, kernel, iterations=2)

    # Find contours of individual lights
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate centroids of contours (light positions)
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            light_positions.append((cx, cy))

    # Draw contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Christmas Lights Detection', frame)

    # Display light positions with numbers
    for i, pos in enumerate(light_positions):
        cv2.circle(frame, pos, 5, (255, 0, 0), -1)
        cv2.putText(frame, str(i+1), pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()