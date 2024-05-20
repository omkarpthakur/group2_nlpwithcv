import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np

# Load pre-trained object detection model
object_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
object_detection_model.eval()

# Load pre-trained transformer model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transformer_model = BertModel.from_pretrained('bert-base-uncased')
transformer_model.eval()

# Preprocess image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# Tokenize text description
def tokenize_text(text):
    text_tokens = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
    return text_tokens


# Object detection
def detect_objects(image):
    with torch.no_grad():
        outputs = object_detection_model(image)
        boxes = outputs[0]['boxes']
        labels = outputs[0]['labels']
    return boxes, labels


# Extract features for each object
def extract_features(image, boxes):
    object_features = []
    for box in boxes:
        # Crop object from image based on bounding box
        x1, y1, x2, y2 = map(int, box.tolist())
        cropped_image = image[:, :, y1:y2, x1:x2]

        # Encode cropped object using transformer model
        with torch.no_grad():
            object_output = transformer_model(input_ids=cropped_image)
            pooled_output = object_output['pooler_output']
            object_features.append(pooled_output)
    return object_features


# Concatenate object features and textual encodings
def concatenate_features(object_features, labels):
    combined_features = []
    for i, label in enumerate(labels):
        text = f"Object {label.item()}"
        text_tokens = tokenize_text(text)
        object_feature = object_features[i]
        # Concatenate features
        combined_feature = torch.cat((object_feature, text_tokens), dim=1)
        combined_features.append(combined_feature)
    return combined_features


# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert frame to tensor and add batch dimension
    frame = frame.transpose(2, 0, 1)  # OpenCV uses BGR, so transpose to convert to RGB
    frame = np.expand_dims(frame, axis=0)
    image = torch.tensor(frame, dtype=torch.float32) / 255.0  # Normalize to [0, 1]

    # Object detection
    boxes, labels = detect_objects(image)

    # Extract features for each object
    object_features = extract_features(image, boxes)

    combined_features = concatenate_features(object_features, labels)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
