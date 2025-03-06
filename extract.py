import torch
from torch import utils
import numpy as np
import torchvision as tv
from torchvision.models import vgg16 as VGG
from torchvision.datasets import Imagenette
import torchvision.transforms as tfs
import torch.nn as nn
from torchsummary import summary
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

class VGG16FeatureExtractor(torch.nn.Module):
    def __init__(self, weights='DEFAULT'):
        super(VGG16FeatureExtractor, self).__init__()
        self.vgg16 = tv.models.vgg16(weights=weights).features
        # Automatically collect indices of all convolutional layers
        self.conv_layers = [i for i, layer in enumerate(self.vgg16) if isinstance(layer, torch.nn.Conv2d)]

    def forward(self, x):
        features = []
        for layer_index, layer in enumerate(self.vgg16):
            x = layer(x)
            if layer_index in self.conv_layers:
                features.append(x)
        return features

    def load_image(self, image_path):
        # Load an image and transform it to the format required by VGG16
        transform = tfs.Compose([
            tfs.Resize(256),
            tfs.CenterCrop(224),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image

    def save_features(self, frame_folder):
        _, save_folder = frame_folder.split('/')
        feature_folder = f'feature_maps/{save_folder}'

        Path(feature_folder).mkdir(parents=True, exist_ok=True)
        
        # Added ./ infront of frame_folder to make it local
        image_filepaths = sorted(glob.glob(f'./{frame_folder}/*'))

        for image_path in image_filepaths:
            
            input_tensor = self.load_image(image_path)
            
            with torch.no_grad():
                features = self.forward(input_tensor)

            # Create a base filename for saving features without the original extension
            base_filename = os.path.split(image_path)[-1].split('.')[0]
            
            filename = f'feature_maps/{save_folder}/{base_filename}.pkl'
            with open(filename, "wb") as f:
                pickle.dump([feature for feature in features], f)
            print(f"Saved all features to {filename}")