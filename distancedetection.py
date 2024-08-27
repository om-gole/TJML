#Om Gole p6 Gabor
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# Load the pre-trained model
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')

# Ensure the model is in evaluation mode
model.eval()

# Load and preprocess the image
image = Image.open('lights.jpg')
# Make sure the image size is divisible by 32
width, height = image.size
width = width - width % 32
height = height - height % 32
image = image.resize((width, height))
transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
input_image = transform(image).unsqueeze(0)

# Perform depth estimation
with torch.no_grad():
    prediction = model(input_image)

# The prediction is a depth map, where each pixel value represents the estimated distance from the camera.
depth_map = prediction.squeeze().cpu().numpy()

# Display the image and let the user select a ROI
cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Select ROI', 600, 600)
roi = cv2.selectROI('Select ROI', np.array(image))

# Crop the depth map to the ROI
roi_depth = depth_map[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

# Calculate the average depth within the ROI
average_depth = np.mean(roi_depth)

print(f'The average estimated distance of the selected object from the camera is: {average_depth} units')