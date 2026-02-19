"""
Web Interface for Neural Style Transfer using Gradio
Allows easy experimentation with different content and style images
"""

import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

class StyleTransferWeb:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing model on {self.device}...")
        # Load VGG19 features
        self.model = models.vgg19(pretrained=True).features.to(self.device).eval()
        
        # Freeze parameters to save memory/computation
        for param in self.model.parameters():
            param.requires_grad_(False)
    
    def load_image(self, img, max_size=400):
        """Load and preprocess image"""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        size = min(max_size, max(img.size))
        
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = transform(img).unsqueeze(0)
        return image.to(self.device)
    
    def im_convert(self, tensor):
        """Convert tensor to numpy array"""
        image = tensor.cpu().clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image = image.clip(0, 1)
        return (image * 255).astype('uint8')
    
    def get_features(self, image):
        """Extract features from VGG layers"""
        layers = {
            '0': 'conv_1', '5': 'conv_2', '10': 'conv_3',
            '19': 'conv_4', '28': 'conv_5'
        }
        
        features = {}
        x = image
        
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                
        return features
    
    def gram_matrix(self, tensor):
        """Calculate gram matrix"""
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram
    
    def transfer(self, content_img, style_img, steps=300, style_weight=1e6, content_weight=1, progress=None):
        """Perform style transfer with progress updates"""
        
        # Preprocess images
        content = self.load_image(content_img)
        style = self.load_image(style_img)
        
        # Initialize target
        target = content.clone().requires_grad_(True)
        
        # Extract features
        content_features = self.get_features(content)
        style_features = self.get_features(style)
        
        # Style gram matrices
        style_grams = {layer: self.gram_matrix(style_features[layer]) 
                      for layer in style_features}
        
        # Optimizer - Increased LR to 0.05 for faster changes
        optimizer = torch.optim.Adam([target], lr=0.05)
        
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        # BLOB FIX: Custom weights to prioritize shapes (conv_4, conv_5) over textures
        layer_weights = {
            'conv_1': 0.1,
            'conv_2': 0.1,
            'conv_3': 1.0,
            'conv_4': 10.0,
            'conv_5': 10.0
        }
        
        # Optimization loop
        for step in range(1, steps + 1):
            target_features = self.get_features(target)
            
            # Content loss
            content_loss = torch.mean(
                (target_features['conv_4'] - content_features['conv_4'])**2
            )
            
            # Style loss with CUSTOM WEIGHTS
            style_loss = 0
            for layer in style_layers:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                style_gram = style_grams[layer]
                
                layer_loss = torch.mean((target_gram - style_gram)**2)
                _, d, h, w = target_feature.shape
                
                # Apply the specific weight for this layer
                style_loss += (layer_loss * layer_weights[layer]) / (d * h * w)
            
            # Total loss using the dynamic content_weight
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update Progress Bar
            if progress:
                progress(step / steps, desc=f"Processing Step {step}/{steps}")

            # Stream the image every 20 steps
            if step % 20 == 0 or step == steps:
                yield self.im_convert(target)

# Initialize model
st_web = StyleTransferWeb()

def process_images(content_img, style_img, iterations, style_strength, content_strength_slider, progress=gr.Progress()):
    """Gradio interface function with Generator"""
    if content_img is None or style_img is None:
        return None
    
    # Increase style weight multiplier
    style_weight = style_strength * 1e8
    
    # Use the content strength from the new slider
    content_weight = content_strength_slider
    
    yield from st_web.transfer(content_img, style_img, steps=int(iterations), 
                             style_weight=style_weight, content_weight=content_weight, progress=progress)

# Create Gradio interface
with gr.Blocks(title="Neural Style Transfer") as demo:
    gr.Markdown("""
    # 🎨 Neural Style Transfer
    Upload a content image and a style image.
    """)
    
    with gr.Row():
        with gr.Column():
            content_input = gr.Image(label="Content Image", type="numpy")
            style_input = gr.Image(label="Style Image", type="numpy")
            
            with gr.Row():
                iterations = gr.Slider(50, 500, value=200, step=50, 
                                     label="Iterations")
                style_strength = gr.Slider(1, 20, value=10, step=1,
                                         label="Style Strength")
                # NEW SLIDER: Content Strength
                content_strength = gr.Slider(0.001, 1.0, value=0.01, step=0.001,
                                           label="Content Strength")
            
            submit_btn = gr.Button("Transfer Style", variant="primary")
        
        with gr.Column():
            output = gr.Image(label="Result (Updates Live)")
    
    submit_btn.click(
        fn=process_images,
        inputs=[content_input, style_input, iterations, style_strength, content_strength],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=True)