# Classes that will be imported into runing code as a regular import statement at the beginning.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Class to find and print layers in neural network
class LayerFinder:
    def __init__(self, model):
        self.model = model

    def print_layers(self):
        print("Layers in the model:")
        for idx, (name, layer) in enumerate(self.model.named_children()):
            print(f"Layer {idx}: {name} -> {layer}")

    def get_layers(self):
        layer_dict = {}
        for idx, (name, layer) in enumerate(self.model.named_children()):
            layer_dict[idx] = layer
        return layer_dict

# Class to apply custom entropy loss based on equation for both convolutional and dense layers
class CustomEntropyLoss(nn.Module):
    def __init__(self, layer_indices, lambdas):
        super(CustomEntropyLoss, self).__init__()
        self.layer_indices = layer_indices
        self.lambdas = lambdas

    def forward(self, model, output, target):
        base_loss = F.cross_entropy(output, target)
        entropy_loss = 0.0
        
        for idx, (name, layer) in enumerate(model.named_children()):
            if idx in self.layer_indices:
                lambda_value = self.lambdas[self.layer_indices.index(idx)]
                
                if isinstance(layer, nn.Linear):
                    W = layer.weight
                    if W.shape[0] == W.shape[1]:
                        det_W = torch.det(W)
                        entropy_loss += -lambda_value * torch.log(torch.abs(det_W) + 1e-6)
                
                elif isinstance(layer, nn.Conv2d):
                    W = layer.weight
                    first_filter = W[0, 0, 0, 0]
                    entropy_loss += -lambda_value * torch.log(torch.abs(first_filter) + 1e-6)

        return base_loss + entropy_loss
