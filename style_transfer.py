"""
Neural Style Transfer Implementation
Transfers artistic style from one image to another using VGG19 and gradient descent
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import copy
import argparse
import os

class StyleTransfer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {device}")
        
        # Load pre-trained VGG1
        self.model = models.vgg19(pretrained=True).features.to(device).eval()
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        # Define layers for style and content
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
    def load_image(self, img_path, max_size=400):
        """Load and preprocess image"""
        image = Image.open(img_path).convert('RGB')
        
        # Resize if too large
        size = min(max_size, max(image.size))
        
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = transform(image).unsqueeze(0)
        return image.to(self.device)
    
    def im_convert(self, tensor):
        """Convert tensor to image for display"""
        image = tensor.cpu().clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image = image.clip(0, 1)
        return image
    
    def get_features(self, image, model):
        """Extract features from specific layers"""
        layers = {
            '0': 'conv_1',
            '5': 'conv_2',
            '10': 'conv_3',
            '19': 'conv_4',
            '28': 'conv_5'
        }
        
        features = {}
        x = image
        
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                
        return features
    
    def gram_matrix(self, tensor):
        """Calculate gram matrix for style representation"""
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram
    def transfer_style(self, content_path, style_path, output_path='output.jpg',
                      steps=2000, style_weight=1e8, content_weight=1,  # Changed 1e6 -> 1e8
                      learning_rate=0.05, show_every=500):             # Changed 0.003 -> 0.05

        """
        Perform style transfer
        
        Args:
            content_path: Path to content image
            style_path: Path to style image
            output_path: Where to save result
            steps: Number of optimization steps
            style_weight: Weight for style loss
            content_weight: Weight for content loss
            learning_rate: Learning rate for optimizer
            show_every: Print loss every N steps
        """
        
        # Load images
        content = self.load_image(content_path)
        style = self.load_image(style_path, max_size=400)
        
        # Initialize target as content image
        target = content.clone().requires_grad_(True)
        
        # Extract features
        content_features = self.get_features(content, self.model)
        style_features = self.get_features(style, self.model)
        
        # Calculate style gram matrices
        style_grams = {layer: self.gram_matrix(style_features[layer]) 
                      for layer in style_features}
        
        # Optimizer
        optimizer = optim.Adam([target], lr=learning_rate)
        
        print(f"\nStarting style transfer for {steps} iterations...")
        print(f"Style weight: {style_weight}, Content weight: {content_weight}\n")
        
        for step in range(1, steps + 1):
            # Extract target features
            target_features = self.get_features(target, self.model)
            
            # Content loss
            content_loss = torch.mean((target_features['conv_4'] - 
                                      content_features['conv_4'])**2)
            
            # Style loss
            style_loss = 0
            for layer in self.style_layers:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                style_gram = style_grams[layer]
                
                layer_style_loss = torch.mean((target_gram - style_gram)**2)
                _, d, h, w = target_feature.shape
                style_loss += layer_style_loss / (d * h * w)
            
            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            # Update target image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Print progress
            if step % show_every == 0 or step == 1:
                print(f"Step {step}/{steps}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Content Loss: {content_loss.item():.4f}")
                print(f"  Style Loss: {style_loss.item():.4f}\n")
        
        # Save result
        result = self.im_convert(target)
        result_img = Image.fromarray((result * 255).astype('uint8'))
        result_img.save(output_path)
        print(f"Saved result to {output_path}")
        
        return result

def main():
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', type=str, required=True, help='Path to content image')
    parser.add_argument('--style', type=str, required=True, help='Path to style image')
    parser.add_argument('--output', type=str, default='output.jpg', help='Output path')
    parser.add_argument('--steps', type=int, default=2000, help='Number of optimization steps')
    # CHANGE 1: Increase style weight from 1e6 to 1e8 (100,000,000)
    parser.add_argument('--style-weight', type=float, default=1e8, help='Style loss weight')
    parser.add_argument('--content-weight', type=float, default=1, help='Content loss weight')
    # CHANGE 2: Increase learning rate from 0.003 to 0.05
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    
    args = parser.parse_args()
    
    # Verify input files exist
    if not os.path.exists(args.content):
        print(f"Error: Content image not found: {args.content}")
        return
    if not os.path.exists(args.style):
        print(f"Error: Style image not found: {args.style}")
        return
    
    # Run style transfer
    st = StyleTransfer()
    st.transfer_style(
        content_path=args.content,
        style_path=args.style,
        output_path=args.output,
        steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        learning_rate=args.lr
    )

if __name__ == '__main__':
    main()
