"""Image backbones that expose feature vectors instead of classification logits."""

import timm
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional
from utils.image_preprocessing import transform_fn


class GC_ViT_224(nn.Module):
    def __init__(self):
        super(GC_ViT_224, self).__init__()
        self.model = timm.create_model(
            'gcvit_base',
            pretrained=True,
            num_classes=0  # Removes classification head
        )
        return
    
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec
    
    def get_transform(self):
        return transform_fn


class ResNet50_Base_224(nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        # Load the pre-trained ResNet50 model
        if weights is not None:
            weights = ResNet50_Weights.IMAGENET1K_V1  # Use the default ImageNet weights
        # Load the pre-trained ResNet50 model
        base_model = resnet50(weights=weights)
        # Remove the final fully connected layer to use the model as a fixed feature extractor
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Remove the last layer (fc layer)
        
        # The output of 'self.features' will be a tensor of shape (batch_size, 2048, 1, 1) from the average pooling layer
        # AdaptiveAvgPool layer added to convert it to (batch_size, 2048) which is easier to use in most tasks
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        # Pass input through the feature layers
        output = self.features(input)
        
        # Apply adaptive pooling to convert the output to shape (batch_size, 2048)
        output = self.adaptive_pool(output)
        output = output.view(output.size(0), -1)  # Flatten the output
        
        return output
    
    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            # Apply the necessary transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return transform(image)
        return transform
