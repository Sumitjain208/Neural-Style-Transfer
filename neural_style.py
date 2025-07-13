import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')
    # Resize image while maintaining aspect ratio
    size = max_size if max(image.size) > max_size else max(image.size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Image post-processing and saving
def post_process(tensor, save_path=None):
    image = tensor.cpu().clone().detach().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    if save_path:
        image.save(save_path)
    return image

# VGG19 model with selected layers for style and content
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
        self.content_layers = ['conv_4_2']
        self.style_layers = ['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_4_1', 'conv_5_1']
        
    def forward(self, x):
        content_features = []
        style_features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.content_layers:
                content_features.append(x)
            if name in self.style_layers:
                style_features.append(x)
        return content_features, style_features

# Gram matrix for style loss
def gram_matrix(tensor):
    batch, channels, height, width = tensor.size()
    features = tensor.view(channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(channels * height * width)

# Loss functions
class StyleContentLoss:
    def __init__(self, content_features, style_features, content_weight=1, style_weight=1e6):
        self.content_features = content_features
        self.style_features = [gram_matrix(f) for f in style_features]
        self.content_weight = content_weight
        self.style_weight = style_weight
        
    def compute_loss(self, input_content_features, input_style_features):
        # Content loss
        content_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for input_f, target_f in zip(input_content_features, self.content_features):
            content_loss = content_loss + torch.mean((input_f - target_f) ** 2)
            
        # Style loss
        style_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for input_f, target_f in zip(input_style_features, self.style_features):
            input_gram = gram_matrix(input_f)
            style_loss = style_loss + torch.mean((input_gram - target_f) ** 2)
            
        # Return the weighted sum as a tensor
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        return total_loss

# Main style transfer function
def style_transfer(content_path, style_path, output_path, num_steps=300, content_weight=1, style_weight=1e6):
    # Load images
    content_img = load_image(content_path)
    style_img = load_image(style_path, max_size=min(content_img.shape[2:]))
    
    # Initialize target image (clone of content image)
    target_img = content_img.clone().requires_grad_(True)
    
    # Initialize model and optimizer
    model = VGG19()
    optimizer = optim.LBFGS([target_img])
    
    # Get target features
    content_features, _ = model(content_img)
    _, style_features = model(style_img)
    loss_fn = StyleContentLoss(content_features, style_features, content_weight, style_weight)
    
    # Optimization loop
    def closure():
        optimizer.zero_grad()
        
        # Get features from current target image
        current_content_features, current_style_features = model(target_img)
        
        # Calculate total loss using our loss function
        loss = loss_fn.compute_loss(current_content_features, current_style_features)
        
        # Compute gradients
        loss.backward()
        
        return loss.item()  # Return float for LBFGS

    for step in range(num_steps):
        optimizer.step(closure)
        if step % 50 == 0:
            current_loss = closure()
            print(f"Step {step}, Loss: {current_loss:.4f}")
    
    # Save and return final image
    final_img = post_process(target_img, output_path)
    return final_img

# Example usage
if __name__ == "__main__":
    # Example image paths (replace with your own images)
    content_path = "content.jpg"  # Your content image
    style_path = "style.jpeg"     # Your style image
    output_path = "output.jpg"    # Output path
    
    # Run style transfer with default weights
    result = style_transfer(
        content_path, 
        style_path, 
        output_path,
        num_steps=300,
        content_weight=1,
        style_weight=1e6
    )
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(Image.open(content_path))
    plt.title("Content Image")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(Image.open(style_path))
    plt.title("Style Image")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(result)
    plt.title("Stylized Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()