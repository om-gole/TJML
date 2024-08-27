#Om Gole p6 Gabor
#Part 5
import cv2
import numpy as np
from math import atan, tan
from numpy.linalg import solve, norm
from numpy import array, cross
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the video
cap = cv2.VideoCapture('tree.mp4')

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 2)

# The video feed is read in as a VideoCapture object
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi corner detection
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Lists to store the coordinates of the lights in each frame
coords1 = []
coords2 = []

while(1):
    ret,frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = map(int, new.ravel())
        c,d = map(int, old.ravel())
        a, b = map(int, new.ravel())
        c, d = map(int, old.ravel())
        frame = cv2.line(frame, (a,b),(c,d), (0,255,0), 2)

        frame = cv2.line(frame, (a,b),(c,d), (0,255,0), 10)  # Increase the thickness of the line to make it longer
        frame = cv2.circle(frame,(a,b),5,(0,0,255),-1)

    # Append the coordinates of the lights to the lists
    coords1.append(good_old)
    coords2.append(good_new)

    cv2.imshow('Frame',frame)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

# Your intersection function and other functions go here

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



def calculate_light_positions(list1, list2, intersection):
    light_positions = []
    paired_positions = []
    for p1, p2 in zip(list1, list2):
        light_position = intersection((p1[0], p1[1]), (1.225, 1.225), 0, [0, .3048, 0], p1, p2)
        light_positions.append(light_position)
        paired_positions.append((p1, p2))
    return light_positions, paired_positions

def round_coordinates(input_list, decimals=5):
    return [np.around(array, decimals) for array in input_list]

# Call your functions with the coordinates of the lights
light_positions, paired_positions = calculate_light_positions(coords1, coords2, intersection)
