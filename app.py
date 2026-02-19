"""
Web Interface for AI 3D Shape Generator
Interactive 3D mesh generation and visualization
"""

import gradio as gr
import trimesh
from shape_generator import AI3DGenerator
import tempfile
import os

# Initialize generator
generator = AI3DGenerator()

def generate_and_export(shape_type, resolution, add_detail, complexity):
    """Generate shape and return files"""
    
    try:
        # 1. Generate mesh
        mesh = generator.generate_shape(
            shape_type=shape_type,
            resolution=int(resolution),
            add_detail=add_detail
        )
        
        # 2. Export GLB for the 3D Viewer (Best for web visibility)
        preview_file = tempfile.NamedTemporaryFile(delete=False, suffix='.glb')
        preview_file.close() # Close to release Windows lock
        mesh.export(preview_file.name, file_type='glb')
        
        # 3. Export OBJ for Download (Best for Blender/Unity)
        download_file = tempfile.NamedTemporaryFile(delete=False, suffix='.obj')
        download_file.close() # Close to release Windows lock
        mesh.export(download_file.name, file_type='obj')
        
        info = f"""
        ### 📊 Shape Statistics
        **Type**: {shape_type.capitalize()}
        **Vertices**: {len(mesh.vertices):,}
        **Faces**: {len(mesh.faces):,}
        **Volume**: {mesh.volume:.2f}
        """
        
        return preview_file.name, download_file.name, info
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="AI 3D Shape Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎲 AI-Powered 3D Shape Generator")
    
    with gr.Row():
        with gr.Column(scale=1):
            shape_type = gr.Dropdown(
                choices=['sphere', 'torus', 'knot', 'shell', 'cube', 'organic'],
                value='sphere', label="Shape Type"
            )
            resolution = gr.Slider(20, 100, value=50, step=5, label="Resolution")
            complexity = gr.Slider(0.5, 2.0, value=1.0, label="Complexity")
            add_detail = gr.Checkbox(label="Add Procedural Detail")
            btn = gr.Button("🎨 Generate Shape", variant="primary")
        
        with gr.Column(scale=2):
            # Interactive 3D Viewer
            preview_3d = gr.Model3D(
                label="3D Preview", 
                clear_color=[0.0, 0.0, 0.0, 0.0], # Transparent background
                interactive=True
            )
            info = gr.Markdown()
            dl = gr.File(label="Download .OBJ")
    
    btn.click(
        fn=generate_and_export,
        inputs=[shape_type, resolution, add_detail, complexity],
        outputs=[preview_3d, dl, info]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")