import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

# Load the pre-trained model
model = MyModel()
model.load_state_dict(torch.load('mnist_model.pt'))

# Define a function to preprocess the image
def preprocess(image):
    # Convert the image to grayscale and apply Gaussian blur and thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize the image to 28x28 and convert it to a PyTorch tensor
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0)
    tensor /= 255.0

    return tensor

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Get the width and height of the video frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the coordinates of the bounding box
bbox_size = (250, 250)
bbox = [(int(width // 2 - bbox_size[0] // 2), int(height // 2 - bbox_size[1] // 2)),
        (int(width // 2 + bbox_size[0] // 2), int(height // 2 + bbox_size[1] // 2))]

# Start the video capture loop
while True:
    # Read a frame from the camera
    _, frame = cap.read()

    # Get the cropped image and preprocess it
    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    img_tensor = preprocess(img_cropped)

    # Make a prediction using the model and get the predicted digit and confidence
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted] * 100

    # Draw the bounding box on the frame
    cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 3)

    # If the confidence is high enough, draw the predicted digit on the frame
    if confidence > 23:
        cv2.putText(frame, str(predicted.item()), (bbox[0][0] + 5, bbox[0][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Display the frames
    cv2.imshow('input', frame)
    #cv2.imshow('cropped', img_cropped)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()