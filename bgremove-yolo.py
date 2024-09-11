import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the COCO dataset class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the input image
image = cv2.imread("medium.jpg")
height, width, channels = image.shape

# Create a blob from the input image
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Perform forward pass to get the detection results
outs = net.forward(output_layers)

# Initialization
class_ids = []
confidences = []
boxes = []

# Loop through detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Filter out weak detections
        if confidence > 0.5:
            # Get the coordinates of the bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate top-left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-max suppression to eliminate redundant overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Create a mask for the detected objects using bounding boxes
mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Initialize the mask with zeros (background)

for i in indices.flatten():
    x, y, w, h = boxes[i]
    # Set the region corresponding to the object in the mask to probable foreground
    mask[y:y+h, x:x+w] = cv2.GC_PR_FGD  # GrabCut probable foreground

# Initialize the background and foreground models needed for GrabCut
bg_model = np.zeros((1, 65), np.float64)
fg_model = np.zeros((1, 65), np.float64)

# Apply GrabCut to refine the mask (it will segment the object from the background)
cv2.grabCut(image, mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)

# Modify the mask: Convert possible foregrounds and definite foregrounds to 1, others to 0
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply the refined mask to the original image to remove the background
result = image * mask2[:, :, np.newaxis]

# Create a white background
background = np.ones_like(image, dtype=np.uint8) * 255

# Combine the foreground (object) with the white background
final_image = background * (1 - mask2[:, :, np.newaxis]) + result

# Save the result image
cv2.imwrite("medium-yolo.jpg", final_image)

# Show the result
cv2.imshow("Original Image", image)
cv2.imshow("Background Removed", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
