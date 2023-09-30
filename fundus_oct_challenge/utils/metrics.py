import torch
import numpy as np
from scipy.spatial import distance
import torch.nn as nn

class MeanEuclideanDistanceEdgeError:
    """GPT-4 wrote this function based on the description given in the challenge, I did not actually check it.
    MED is not a common metric used in image segmentation, we might want to look at this again.

    full disclosure, the prompt was the following:

    I want to write a custom metric based on this description:
    In addition, we use Euclidean distance to measure the error of the edge of the segmentation result and the gold standard. Specifically, we first traverse each pixel on the edge of the segmentation result, and calculate the Euclidean distance from each pixel to the nearest pixel on the gold standard edge. Then the sum of the above Euclidean distances was averaged based on the number of pixels on the edge of the segmentation result:

    the formula given is:
    $MED = \frac{1}{N} \sum_{i=1}^N \sqrt{(x_i - x-x_i^0)^2 + (y_i - y_i^0)^2}$

    where, N is the number of pixels on the edge of the segmentation result, $(x_i, y_i)$ is the ith pixel on the edge of the segmentation and $(x_i^0, y_i^0)$ is the nearest pixel on the gold standard edge to $(x_i, y_i)$

    please write this metric for me
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_distance = 0.0
        self.num_edges = 0

    def __call__(self, preds, targets):
        # Assuming preds and targets are 4D tensors (B, C, H, W) with C=1
        for pred, target in zip(preds, targets):
            # 1. Detect edges
            pred_edges = self.detect_edges(pred.squeeze(0))
            target_edges = self.detect_edges(target.squeeze(0))

            # 2. For each edge pixel in pred, compute distance to the nearest edge pixel in target
            pred_edge_positions = torch.nonzero(pred_edges, as_tuple=False)
            target_edge_positions = torch.nonzero(target_edges, as_tuple=False).cpu().numpy()
            
            for position in pred_edge_positions:
                dists = distance.cdist([position.cpu().numpy()], target_edge_positions, 'euclidean')
                nearest_distance = dists.min()
                self.total_distance += nearest_distance
                self.num_edges += 1

        return self.total_distance / max(self.num_edges, 1)

    def compute(self):
        return self.total_distance / max(self.num_edges, 1)

    def detect_edges(self, img):
        img = img.float()  # Convert img to float tensor
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=img.device)
        
        G_x = torch.nn.functional.conv2d(img.unsqueeze(0).unsqueeze(0), sobel_x.float().unsqueeze(0).unsqueeze(0))
        G_y = torch.nn.functional.conv2d(img.unsqueeze(0).unsqueeze(0), sobel_y.float().unsqueeze(0).unsqueeze(0))
        
        G = torch.sqrt(G_x**2 + G_y**2)
        edges = (G > 0).squeeze()
        
        return edges

class DiceCoefficient(nn.Module):
    def __init__(self, num_classes, class_weights=None, epsilon=1e-7):
        super(DiceCoefficient, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
        # If no class weights are provided, assign equal weights to all classes
        if class_weights is None:
            class_weights = torch.ones(num_classes)
        
        self.register_buffer('class_weights', class_weights)

    def forward(self, input_tensor, target_tensor):
        # Ensure the tensors are on the same device
        input_tensor = input_tensor.to(target_tensor.device)

        # Convert the input tensor to the same shape as target tensor
        _, preds = torch.max(input_tensor, dim=1)

        dice_values = torch.zeros(self.num_classes).to(target_tensor.device)

        for class_idx in range(self.num_classes):
            input_i = (preds == class_idx).float()
            target_i = (target_tensor == class_idx).float()

            intersection = (input_i * target_i).sum()
            union = input_i.sum() + target_i.sum()

            dice_values[class_idx] = (2. * intersection) / (union + self.epsilon)

        # Weighted mean
        return (dice_values * self.class_weights).sum() / self.class_weights.sum()
