import cv2
import numpy as np
import torch
from torchvision import models, transforms

def remove_background(image_path, output_path, threshold=0.5):
    # Load pre-trained Mask R-CNN model using weights
    weights = models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = models.detection.maskrcnn_resnet50_fpn(weights=weights)
    model.eval()

    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transform image
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image_rgb)

    # Perform inference
    with torch.no_grad():
        outputs = model([image_tensor])

    # Extract masks and scores
    masks = outputs[0]['masks']
    scores = outputs[0]['scores']

    # Create an empty mask for the foreground
    final_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Combine masks with scores above the threshold
    for mask, score in zip(masks, scores):
        if score > threshold:
            mask = mask[0].mul(255).byte().cpu().numpy()
            final_mask = np.maximum(final_mask, mask)

    # Apply morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    # Ensure the mask is binary
    _, binary_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

    # Create a white background image
    white_background = np.ones_like(image) * 255

    # Combine the original image with the white background using the binary mask
    image_with_white_bg = np.where(binary_mask[..., None], image, white_background)

    # Save the resulting image
    cv2.imwrite(output_path, image_with_white_bg)

# Usage
remove_background("medium.jpg", "py-medium.png")
