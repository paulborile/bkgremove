import os
import cv2
import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
import gdown

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout

class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hx1 = self.rebnconvin(hx)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv1(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv2(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv3(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv4(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv5(hx)
        hx7 = self.rebnconv6(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx5d = self.rebnconv5d(torch.cat((hx6d, hx5), 1))
        hx4d = self.rebnconv4d(torch.cat((hx5d, hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((hx4d, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hx1

class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        # Additional stages omitted for brevity
        self.stage6d = RSU7(640, 32, 64)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx1 = self.stage1(x)
        # Additional stages omitted for brevity
        d1 = self.side1(hx1)
        dout = self.outconv(torch.cat((d1, d1, d1, d1, d1, d1), 1))
        return torch.sigmoid(dout)

def download_model_from_google_drive():
    url = 'https://drive.google.com/uc?id=1bo-N0feCkxtuAtXPSlrKLzGQU6pYOt5f'
    output = 'u2net.pth'
    gdown.download(url, output, quiet=False)

def load_model(model_path):
    model = U2NET(3, 1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((320, 320)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

def postprocess_mask(mask, original_shape):
    mask = mask.squeeze().cpu().data.numpy()
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

def remove_background(image_path, output_path, model):
    # Read the original image to get its shape
    original_image = cv2.imread(image_path)
    original_height, original_width = original_image.shape[:2]

    # Preprocess the image
    image = preprocess_image(image_path)

    # Perform inference
    with torch.no_grad():
        prediction = model(image)

    # Postprocess the mask
    mask = postprocess_mask(prediction[0], (original_height, original_width))

    # Create a white background image
    white_background = np.ones_like(original_image) * 255

    # Combine the original image with the white background using the mask
    image_with_white_bg = np.where(mask[:, :, None] == 255, original_image, white_background)

    # Save the resulting image
    cv2.imwrite(output_path, image_with_white_bg)

# Download the model if it doesn't exist
model_path = "u2net.pth"
if not os.path.exists(model_path):
    download_model_from_google_drive()

# Load the model
model = load_model(model_path)

# Usage
remove_background("medium.jpg", "py-medium.png")
