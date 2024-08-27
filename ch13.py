from torchvision import models
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load a pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Define a function to apply semantic segmentation to an image
def apply_semantic_segmentation(image_path):
    # Load the image
    input_image = Image.open(image_path)
    
    # Define the standard transformation
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Preprocess the image
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Check if a GPU is available and if so, use it
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    # Apply the model to the image
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    # The output is a probability map, take the maximum value to get the predicted class
    output_predictions = output.argmax(0)
    
    # Plot the semantic segmentation
    plt.imshow(output_predictions.cpu().numpy())
    plt.show()

# Example usage with a hypothetical image path
apply_semantic_segmentation('FEIN.jpg')

