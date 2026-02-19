"""
Web Interface for AI 3D Shape Generator
Interactive 3D mesh generation and visualization
"""

import gradio as gr
import numpy as np
import trimesh
from shape_generator import AI3DGenerator
import tempfile
import os

# Initialize generator
generator = AI3DGenerator()

def generate_and_export(shape_type, resolution, add_detail, complexity):
    """Generate shape and return file for download"""
    
    try:
        # Generate mesh
        mesh = generator.generate_shape(
            shape_type=shape_type,
            resolution=int(resolution),
            add_detail=add_detail
        )
        
        # Export to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.obj')
        mesh.export(temp_file.name)
        
        # Generate simple preview image
        scene = mesh.scene()
        png = scene.save_image(resolution=[400, 400])
        
        # Save preview
        preview_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        with open(preview_file.name, 'wb') as f:
            f.write(png)
        
        info = f"""
        **Generated Shape**: {shape_type}
        **Vertices**: {len(mesh.vertices):,}
        **Faces**: {len(mesh.faces):,}
        **Surface Area**: {mesh.area:.2f}
        **Volume**: {mesh.volume:.2f}
        """
        
        return preview_file.name, temp_file.name, info
        
    except Exception as e:
        error_msg = f"Error generating shape: {str(e)}"
        print(error_msg)
        return None, None, error_msg

def create_3d_viewer_html(obj_path):
    """Create HTML with Three.js viewer"""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; overflow: hidden; }}
            #canvas {{ width: 100%; height: 600px; }}
        </style>
    </head>
    <body>
        <div id="canvas"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 600, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            
            renderer.setSize(window.innerWidth, 600);
            document.getElementById('canvas').appendChild(renderer.domElement);
            
            // Lighting
            const light1 = new THREE.DirectionalLight(0xffffff, 1);
            light1.position.set(1, 1, 1);
            scene.add(light1);
            
            const light2 = new THREE.DirectionalLight(0xffffff, 0.5);
            light2.position.set(-1, -1, -1);
            scene.add(light2);
            
            scene.add(new THREE.AmbientLight(0x404040));
            
            // Load geometry (simplified - showing a sphere as example)
            const geometry = new THREE.SphereGeometry(1, 32, 32);
            const material = new THREE.MeshPhongMaterial({{ 
                color: 0x2196F3,
                shininess: 100,
                specular: 0x111111
            }});
            const mesh = new THREE.Mesh(geometry, material);
            scene.add(mesh);
            
            camera.position.z = 3;
            
            // Animation
            function animate() {{
                requestAnimationFrame(animate);
                mesh.rotation.x += 0.005;
                mesh.rotation.y += 0.01;
                renderer.render(scene, camera);
            }}
            animate();
            
            // Handle resize
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / 600;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, 600);
            }});
        </script>
    </body>
    </html>
    """
    return html

# Create Gradio interface
with gr.Blocks(title="AI 3D Shape Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎲 AI-Powered 3D Shape Generator
    ### Generate 3D meshes using neural networks and procedural techniques
    
    Create complex 3D shapes with customizable parameters. Download as .OBJ files 
    compatible with Blender, Unity, Unreal Engine, and other 3D software.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Shape Parameters")
            
            shape_type = gr.Dropdown(
                choices=['sphere', 'torus', 'knot', 'shell', 'cube', 'organic'],
                value='sphere',
                label="Shape Type",
                info="Select base shape geometry"
            )
            
            resolution = gr.Slider(
                minimum=20,
                maximum=100,
                value=50,
                step=5,
                label="Resolution",
                info="Higher = more detailed (slower)"
            )
            
            complexity = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Complexity",
                info="Shape variation parameter"
            )
            
            add_detail = gr.Checkbox(
                value=False,
                label="Add Procedural Detail",
                info="Add noise-based surface details"
            )
            
            generate_btn = gr.Button("🎨 Generate Shape", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 👁️ Preview & Download")
            
            preview_img = gr.Image(label="Shape Preview", type="filepath")
            
            mesh_info = gr.Markdown("Generate a shape to see details...")
            
            output_file = gr.File(label="Download 3D Model (.OBJ)")
    
    # Connect button
    generate_btn.click(
        fn=generate_and_export,
        inputs=[shape_type, resolution, add_detail, complexity],
        outputs=[preview_img, output_file, mesh_info]
    )
    
    gr.Markdown("""
    ---
    ## 📚 Shape Types
    
    - **Sphere**: Basic spherical geometry
    - **Torus**: Donut-shaped surface
    - **Knot**: Mathematical trefoil knot
    - **Shell**: Nautilus-like spiral shell
    - **Cube**: Box with optional organic deformation
    - **Organic**: Sphere with procedural noise
    
    ## 🎯 Technical Details
    
    **Generation Methods:**
    - **Parametric Surfaces**: Mathematical equations for smooth shapes
    - **Signed Distance Functions**: Implicit surface representation
    - **Neural Implicit Networks**: ML-based shape encoding (advanced)
    - **Procedural Noise**: Perlin-like noise for organic details
    
    **Mesh Processing:**
    - Delaunay triangulation for surface reconstruction
    - Laplacian smoothing for organic shapes
    - Duplicate removal and cleanup
    
    ## 💡 Usage Tips
    
    1. **Start with low resolution** (30-50) for quick previews
    2. **Increase resolution** (70-100) for final exports
    3. **Add detail** for more interesting, organic-looking shapes
    4. **Adjust complexity** to control shape variations
    
    ## 🔧 Export Formats
    
    Generated meshes are exported as .OBJ files, which are compatible with:
    - **Blender** (3D modeling)
    - **Unity** (game development)
    - **Unreal Engine** (game development)
    - **Maya, 3ds Max** (professional 3D)
    - **MeshLab** (mesh processing)
    
    ## 🎓 For Portfolio/CV
    
    This project demonstrates:
    - **3D Geometry**: Parametric surfaces, implicit functions, mesh generation
    - **Computer Graphics**: Rendering, triangulation, surface reconstruction
    - **Neural Networks**: Implicit shape representation with MLPs
    - **Computational Geometry**: SDF evaluation, marching cubes concepts
    
    ### Applications
    - Procedural content generation for games
    - 3D asset creation for VR/AR
    - Geometric deep learning research
    - Automated CAD modeling
    
    ---
    **Author**: Prisati Bhattacharjee  
    **Tech Stack**: Python, PyTorch, Trimesh, NumPy, SciPy
    """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
