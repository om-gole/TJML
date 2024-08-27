# # # FOR IMAGE
import cv2
import numpy as np

# This will be the list of polygon vertices
polygons = []
current_polygon = []

def draw_polygons(image, polygons):
    for polygon in polygons:
        if len(polygon) > 1:
            cv2.polylines(image, [np.array(polygon)], True, (0,255,0), 1)
        for vertex in polygon:
            cv2.circle(image, tuple(vertex), 2, (0,0,255), -1)

def mouse_event(event, x, y, flags, param):
    # If the left mouse button was clicked, record the position
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([x, y])
    # If the right mouse button was clicked, remove the last vertex
    elif event == cv2.EVENT_RBUTTONDOWN and current_polygon:
        current_polygon.pop()

def detect_lights(frame):
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Threshold the image to reveal the light regions in the blurred image
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    # Perform a series of dilations to remove any small blobs of noise from the thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    # Find contours in the mask
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Load the image
image = cv2.imread('lightson.png')
clone = image.copy()

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_event)
count = 0
while True:
    # Display the image
    image = clone.copy()
    draw_polygons(image, [current_polygon])
    draw_polygons(image, polygons)
    cv2.imshow("image", image)

    key = cv2.waitKey(1) & 0xFF

    # If the 'r' key is pressed, reset the drawing
    if key == ord("r"):
        current_polygon = []
    # If the 'c' key is pressed, close the polygon
    elif key == ord("c") and len(current_polygon) > 2:
        polygons.append(current_polygon)
        current_polygon = []
    # If the 'd' key is pressed, detect lights in the last polygon
    elif key == ord("d") and polygons:
        mask = np.zeros(clone.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [np.array(polygons[-1])], -1, (255), thickness=cv2.FILLED)
        selected_area = cv2.bitwise_and(clone, clone, mask=mask)
        detected_lights = detect_lights(selected_area)
        for light in detected_lights:
            print(light)
    # If the 's' key is pressed, save the image of the detected lights within the polygon
    elif key == ord("s") and polygons:
        mask = np.zeros(clone.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [np.array(polygons[-1])], -1, (255), thickness=cv2.FILLED)
        selected_area = cv2.bitwise_and(clone, clone, mask=mask)
        detected_lights = detect_lights(selected_area)
        for light in detected_lights:
            cv2.drawContours(selected_area, [light], -1, (0, 255, 0), 3)
            count = count+1

        cv2.imwrite('detected_lights.jpg', selected_area)
    # If the 'q' key is pressed, break from the loop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
print(count)

# import cv2
# import numpy as np

# # This will be the list of polygon vertices
# polygons_on = []
# polygons_off = []
# current_polygon = []

# def draw_polygons(image, polygons):
#     for polygon in polygons:
#         if len(polygon) > 1:
#             cv2.polylines(image, [np.array(polygon)], True, (0, 255, 0), 1)
#         for vertex in polygon:
#             cv2.circle(image, tuple(vertex), 2, (0, 0, 255), -1)

# def mouse_event(event, x, y, flags, param):
#     # If the left mouse button was clicked, record the position
#     if event == cv2.EVENT_LBUTTONDOWN:
#         current_polygon.append([x, y])
#     # If the right mouse button was clicked, remove the last vertex
#     elif event == cv2.EVENT_RBUTTONDOWN and current_polygon:
#         current_polygon.pop()

# colors = {
#     'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
#     'green': (np.array([35, 50, 50]), np.array([90, 255, 255])),
#     'blue': (np.array([100, 100, 100]), np.array([130, 255, 255])),
#     'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
#     'orange': (np.array([10, 100, 100]), np.array([20, 255, 255])),
#     'purple': (np.array([130, 50, 50]), np.array([160, 255, 255])),
#     'cyan': (np.array([85, 100, 100]), np.array([100, 255, 255])),
# }

# def detect_lights(frame, max_radius=10):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
#     thresh = cv2.dilate(thresh, None, iterations=2)
#     contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filtered_contours = [cnt for cnt in contours if cv2.minEnclosingCircle(cnt)[1] < max_radius]
#     return filtered_contours

# def subtract_lights(image_with_lights_on, image_with_lights_off, distance_threshold=10):
#     # Load the images
#     lights_on = cv2.imread(image_with_lights_on)
#     lights_off = cv2.imread(image_with_lights_off)

#     # Ensure the images are valid
#     if lights_on is None or lights_off is None:
#         print("Error: Unable to read input images.")
#         return

#     # Detect lights in the images
#     detected_lights_on = detect_lights(lights_on)
#     detected_lights_off = detect_lights(lights_off)

#     # Create masks for lights_on and lights_off
#     mask_on = np.zeros_like(lights_on)
#     mask_off = np.zeros_like(lights_off)

#     for light_on in detected_lights_on:
#         cv2.drawContours(mask_on, [light_on], -1, (255, 255, 255), thickness=cv2.FILLED)

#     for light_off in detected_lights_off:
#         cv2.drawContours(mask_off, [light_off], -1, (255, 255, 255), thickness=cv2.FILLED)

#     # Subtract lights detected in lights_off from lights_on
#     result_mask = cv2.subtract(mask_on, mask_off)

#     # Apply the result mask to lights_on
#     result_image = cv2.bitwise_and(lights_on, result_mask)

#     # Save the result
#     cv2.imwrite('result_image.jpg', result_image)



# # Load the images
# image_on = cv2.imread('lightson.png')
# image_off = cv2.imread('lightsoff.png')

# # Create clones for drawing polygons
# clone_on = image_on.copy()
# clone_off = image_off.copy()

# cv2.namedWindow("image_on")
# cv2.setMouseCallback("image_on", mouse_event)

# while True:
#     # Display the image with lights on
#     image_on = clone_on.copy()
#     draw_polygons(image_on, [current_polygon])
#     draw_polygons(image_on, polygons_on)
#     cv2.imshow("image_on", image_on)

#     key = cv2.waitKey(1) & 0xFF

#     # Your existing key handling...
#     # If the 'r' key is pressed, reset the drawing
#     if key == ord("r"):
#         current_polygon = []
#     # If the 'c' key is pressed, close the polygon for image with lights on
#     elif key == ord("c") and len(current_polygon) > 2:
#         polygons_on.append(current_polygon)
#         current_polygon = []

#     # If the 's' key is pressed, save the image of the detected lights within the polygon for image with lights on
#     elif key == ord("s") and polygons_on:
#         mask_on = np.zeros(clone_on.shape[:2], dtype=np.uint8)
#         cv2.drawContours(mask_on, [np.array(polygons_on[-1])], -1, (255), thickness=cv2.FILLED)
#         selected_area_on = cv2.bitwise_and(clone_on, clone_on, mask=mask_on)
#         detected_lights_on = detect_lights(selected_area_on)
#         for light in detected_lights_on:
#             cv2.drawContours(selected_area_on, [light], -1, (255, 0, 0), 3)
#         cv2.imwrite('selected_area_with_lights_on.jpg', selected_area_on)
#         print("Saved image with lights on.")

#     # If the 'q' key is pressed, break from the loop
#     elif key == ord("q"):
#         break

# cv2.destroyAllWindows()

# cv2.namedWindow("image_off")
# cv2.setMouseCallback("image_off", mouse_event)

# while True:
#     # Display the image with lights off
#     image_off = clone_off.copy()
#     draw_polygons(image_off, [current_polygon])
#     draw_polygons(image_off, polygons_off)
#     cv2.imshow("image_off", image_off)

#     key = cv2.waitKey(1) & 0xFF

#     # Your existing key handling...
#     # If the 'r' key is pressed, reset the drawing
#     if key == ord("r"):
#         current_polygon = []
#     # If the 'c' key is pressed, close the polygon for image with lights off
#     elif key == ord("c") and len(current_polygon) > 2:
#         polygons_off.append(current_polygon)
#         current_polygon = []

#     # If the 's' key is pressed, save the image of the detected lights within the polygon for image with lights off
#     elif key == ord("s") and polygons_off:
#         mask_off = np.zeros(clone_off.shape[:2], dtype=np.uint8)
#         cv2.drawContours(mask_off, [np.array(polygons_off[-1])], -1, (255), thickness=cv2.FILLED)
#         selected_area_off = cv2.bitwise_and(clone_off, clone_off, mask=mask_off)
#         detected_lights_off = detect_lights(selected_area_off)
#         for light in detected_lights_off:
#             cv2.drawContours(selected_area_off, [light], -1, (0, 0, 255), 3)
#         cv2.imwrite('selected_area_with_lights_off.jpg', selected_area_off)
#         print("Saved image with lights off.")

#     # If the 'q' key is pressed, break from the loop
#     elif key == ord("q"):
#         break

# cv2.destroyAllWindows()

# # Call the subtract_lights function
# # Call the subtract_lights function
# subtract_lights('selected_area_with_lights_on.jpg', 'selected_area_with_lights_off.jpg')


# # Display the result

# initial_image = cv2.imread('selected_area_with_lights_on.jpg')
# final_image = cv2.imread('selected_area_with_lights_off.jpg')
# result_image = cv2.imread('lightstroll.png')

# # Ensure all images are the same size
# height = min(initial_image.shape[0], final_image.shape[0], result_image.shape[0])
# width = min(initial_image.shape[1], final_image.shape[1], result_image.shape[1])

# initial_image = cv2.resize(initial_image, (width, height))
# final_image = cv2.resize(final_image, (width, height))
# result_image = cv2.resize(result_image, (width, height))

# # Concatenate the images
# concatenated_image = np.concatenate((initial_image, final_image, result_image), axis=1)

# # Save the concatenated image
# cv2.imwrite('concatenated_image.jpg', concatenated_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()