#Om Gole p6 Gabor
#Parts 1,2,3,4
#depth from camera placement
#x/y from light detection
#one picture with set D movement to the right
#triangulation

#no data entry for how far you move + detecting the light

#Wants pictures of the detected lights
import cv2
import numpy as np
from math import sin, cos, tan, atan
from numpy import array, cross
from numpy.linalg import solve, norm
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# This will be the list of polygon vertices
polygons = []
current_polygon = []

def cartesian_to_spherical(p, phi, theta):
    return p * cos(phi) * cos(theta), p * cos(phi) * sin(theta), p * sin(phi)

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
    
    # Find the center of each contour and add it to the list of light coordinates
    light_coords = []
    for contour in contours:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        light_coords.append((cX, cY))
    
    return light_coords

image = cv2.imread('lps1.jpg')
clone = image.copy()
clone = cv2.resize(image, (960, 600))
height, width = image.shape[:2]
center_preshift = (width // 2, height // 2)



cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_event)

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
        # Highlight detected lights
        for light in detected_lights:
            cv2.circle(selected_area, light, 5, (0, 255, 0), 3)
        # Save the image
        cv2.imwrite('detected_lights_PRE.jpg', selected_area)
        
    # If the 'q' key is pressed, break from the loop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
print("Preshift Detected Lights: ", detected_lights)
n_init = len(detected_lights)
print(len(detected_lights))
preshift_lights = detected_lights

#PRE TO POST

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
    
    # Find the center of each contour and add it to the list of light coordinates
    light_coords = []
    for contour in contours:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        light_coords.append((cX, cY))
    
    return light_coords

image = cv2.imread('lps2.jpg')
height, width = image.shape[:2]
center_postshift = (width // 2, height // 2)

clone = image.copy()
clone = cv2.resize(image, (960, 600))

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_event)

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
        # Highlight detected lights
        for light in detected_lights:
            cv2.circle(selected_area, light, 5, (0, 255, 0), 3)
        # Save the image
        cv2.imwrite('detected_lights_POST.jpg', selected_area)
        
    # If the 'q' key is pressed, break from the loop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
print("Postshift Detected Lights: ", detected_lights)
print(len(detected_lights))
postshift_lights = detected_lights

def cartesian_to_spherical(r, theta, phi):
    return [r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)]

def find_intersection(dimensions, field_of_view, altitude, offset, vector1, vector2):
    width, height = dimensions
    horizontal_fov, vertical_fov = field_of_view

    distance1 = width / tan(horizontal_fov / 2)
    distance2 = height / tan(vertical_fov / 2)

    vector1_x, vector1_y = vector1
    vector2_x, vector2_y = vector2

    vector1_x = vector1_x - width / 2
    vector1_y = -vector1_y + height / 2
    vector2_x = vector2_x - width / 2
    vector2_y = -vector2_y + height / 2

    angle1 = atan(vector1_x / distance1)
    angle2 = atan(vector1_y / distance2)
    angle3 = atan(vector2_x / distance1)
    angle4 = atan(vector2_y / distance2)

    initial_position = array([0, 0, altitude])
    final_position = initial_position + array(offset)

    direction1 = array(cartesian_to_spherical(1, angle2, angle1))
    direction2 = array(cartesian_to_spherical(1, angle4, angle3))
    direction_cross = cross(direction2, direction1)
    direction_cross /= norm(direction_cross)

    right_hand_side = final_position - initial_position
    left_hand_side = array([direction1, -direction2, direction_cross]).T
    solution = solve(left_hand_side, right_hand_side)

    return (initial_position + solution[0] * direction1 + final_position + solution[1] * direction2) / 2

# def process_coordinate_lists(list1, list2):
#     # Sort the lists in descending order based on the y-coordinate
#     list1.sort(key=lambda coord: coord[1], reverse=True)
#     list2.sort(key=lambda coord: coord[1], reverse=True)

#     # Make the lists the same size
#     if len(list1) > len(list2):
#         list1 = list1[:len(list2)]
#     elif len(list2) > len(list1):
#         list2 = list2[:len(list1)]

#     return list1, list2

def process_coordinate_lists(list1, list2, n_clusters=2):
    # Extract y-coordinates
    y1 = np.array([coord[1] for coord in list1])
    y2 = np.array([coord[1] for coord in list2])

    # Reshape for clustering
    y1 = y1.reshape(-1, 1)
    y2 = y2.reshape(-1, 1)

    # # Apply KMeans clustering
    # kmeans1 = KMeans(n_clusters=n_clusters, random_state=0).fit(y1)
    # kmeans2 = KMeans(n_clusters=n_clusters, random_state=0).fit(y2)
    kmeans1 = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=0).fit(y1)
    kmeans2 = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=0).fit(y2)


    # Get labels for each point
    labels1 = kmeans1.labels_
    labels2 = kmeans2.labels_

    # Filter points in each list based on cluster labels
    filtered_list1 = [coord for coord, label in zip(list1, labels1) if label in labels2]
    filtered_list2 = [coord for coord, label in zip(list2, labels2) if label in labels1]

    return filtered_list1, filtered_list2

(x,y) = center_preshift
# p1 = preshift_lights[0]
# p2 = postshift_lights[0]


# light_position = intersection((x, y), (1.225, 1.225), 0, [0, .3048, 0], p1, p2)

#for x in preshift_lights.length:

#output is Depth, Offset from center of first pic, then height from first pic
#print(light_position)

coords1, coords2 = process_coordinate_lists(preshift_lights, postshift_lights)

# Plot the original lists
plt.scatter(*zip(*preshift_lights), color='blue', label='List 1')
plt.scatter(*zip(*postshift_lights), color='red', label='List 2')

# Plot the filtered lists
plt.scatter(*zip(*coords1), color='cyan', label='Filtered List 1')
plt.scatter(*zip(*coords2), color='magenta', label='Filtered List 2')

# Add a legend
plt.legend()

# Show the plot
plt.show()

def calculate_light_positions(list1, list2, intersection):
    light_positions = []
    paired_positions = []
    for p1, p2 in zip(list1, list2):
        light_position = find_intersection((p1[0], p1[1]), (1.225, 1.225), 0, [0, .3048, 0], p1, p2)
        light_positions.append(light_position)
        paired_positions.append((p1, p2))
    return light_positions, paired_positions

def round_coordinates(input_list, decimals=5):
    return [np.around(array, decimals) for array in input_list]

light_positions, paired_positions = calculate_light_positions(coords1, coords2, find_intersection)
light_positions = round_coordinates(light_positions)
print("The Paired Positions: ",paired_positions)
print("Positions are calculated in feet and displayed in [x,y,z]", light_positions)
print(len(light_positions))