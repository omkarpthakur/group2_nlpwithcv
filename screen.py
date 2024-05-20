import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
from PIL import Image
import numpy as np

# Load pre-trained DenseNet model
model = densenet121(pretrained=True)
model.eval()

# Load class labels for ImageNet
with open("imagenet-classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(degrees=10),  # Randomly rotate the image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to detect objects from camera feed
def detect_objects():
    # Open camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform preprocessing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame_pil = Image.fromarray(frame_rgb)  # Convert to PIL Image
        input_img = transform(frame_pil).unsqueeze(0)  # Apply transformations

        with torch.no_grad():
            output = model(input_img)
            _, predicted_idx = torch.max(output, 1)

        predicted_label = classes[predicted_idx]

        cv2.putText(frame, predicted_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Object Detection', cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_detected_objects():
    # Open camera
    cap = cv2.VideoCapture(0)

    detected_labels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform preprocessing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame_pil = Image.fromarray(frame_rgb)  # Convert to PIL Image
        input_img = transform(frame_pil).unsqueeze(0)  # Apply transformations

        with torch.no_grad():
            output = model(input_img)
            _, predicted_idx = torch.max(output, 1)

        predicted_label = classes[predicted_idx]
        detected_labels.append(predicted_label)

        cv2.putText(frame, predicted_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Object Detection', cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert detected labels to a single string
    detected_labels_str = ', '.join(detected_labels)

    return detected_labels_str
#ut1
#detect_objects()
#ut2
"""g = get_detected_objects()
print(g)"""
"""detect_objects()"""


