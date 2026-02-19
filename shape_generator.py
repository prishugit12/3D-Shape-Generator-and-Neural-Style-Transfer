"""
AI-Powered 3D Shape Generator
Generate 3D meshes using neural networks and procedural techniques
"""

import torch
import torch.nn as nn
import numpy as np
import trimesh
from scipy.spatial import Delaunay
import argparse
import os

class ImplicitShapeNetwork(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )
        
    def forward(self, coords):
        return self.net(coords)

class AI3DGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {device}")
        self.implicit_net = ImplicitShapeNetwork().to(device)
        self.shape_params = {
            'sphere': {'complexity': 1.0},
            'cube': {'complexity': 0.3},
            'torus': {'complexity': 1.5},
            'knot': {'complexity': 1.0},
            'shell': {'complexity': 1.0},
            'organic': {'complexity': 1.8}
        }
    
    def sdf_sphere(self, points, center=[0, 0, 0], radius=1.0):
        points = np.array(points)
        return np.linalg.norm(points - np.array(center), axis=-1) - radius
    
    def sdf_box(self, points, size=[1, 1, 1]):
        points = np.abs(np.array(points))
        q = points - np.array(size)
        return (np.linalg.norm(np.maximum(q, 0), axis=-1) + 
                np.minimum(np.max(q, axis=-1), 0))
    
    def add_noise(self, points, sdf_values, noise_scale=0.1, frequency=3.0):
        points = np.array(points)
        noise = np.zeros(len(points))
        for octave in range(3):
            scale = frequency * (2 ** octave)
            octave_noise = (np.sin(points[:, 0] * scale) * np.cos(points[:, 1] * scale) * np.sin(points[:, 2] * scale))
            noise += octave_noise / (2 ** octave)
        return sdf_values + noise * noise_scale
    
    def marching_cubes_basic(self, sdf_func, resolution=64, bounds=2.0):
        print(f"Generating mesh with resolution {resolution}^3...")
        x = np.linspace(-bounds, bounds, resolution)
        y = np.linspace(-bounds, bounds, resolution)
        z = np.linspace(-bounds, bounds, resolution)
        grid_points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        sdf_values = sdf_func(grid_points)
        
        surface_mask = np.abs(sdf_values) < 0.1
        surface_points = grid_points[surface_mask]
        
        if len(surface_points) < 4:
            # Fallback for empty meshes
            return self.generate_parametric_shape('sphere', resolution)

        try:
            norms = np.linalg.norm(surface_points, axis=1, keepdims=True)
            projected = surface_points / (norms + 1e-6)
            hull = Delaunay(projected[:, :2])
            return surface_points, hull.simplices
        except:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(surface_points)
            return surface_points, hull.simplices
    
    def generate_parametric_shape(self, shape_type='sphere', resolution=50, complexity=1.0):
        # 1. GENERATE GRID
        if shape_type == 'knot':
            # Knots need to be longer, so we use a different aspect ratio for t/s
            u_res = resolution * 3
            v_res = resolution
        else:
            u_res = resolution
            v_res = resolution

        u = np.linspace(0, 2 * np.pi, u_res)
        v = np.linspace(0, 2 * np.pi, v_res)
        u, v = np.meshgrid(u, v)
        u = u.flatten()
        v = v.flatten()
        
        # 2. CALCULATE COORDINATES
        if shape_type == 'sphere':
            v_sphere = np.linspace(0, np.pi, v_res) # Fix sphere pole overlap
            u_grid, v_grid = np.meshgrid(np.linspace(0, 2*np.pi, u_res), v_sphere)
            u, v = u_grid.flatten(), v_grid.flatten()
            x = np.sin(v) * np.cos(u)
            y = np.sin(v) * np.sin(u)
            z = np.cos(v)
            
        elif shape_type == 'torus':
            R, r = 1.0, 0.3 * complexity
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            
        elif shape_type == 'knot':
            # Trefoil knot parameters
            t = u # Treat u as t (length)
            s = v # Treat v as s (tube)
            r = 0.5 + 0.3 * np.cos(3 * t)
            x = r * np.cos(2 * t) + 0.1 * complexity * np.cos(s)
            y = r * np.sin(2 * t) + 0.1 * complexity * np.sin(s)
            z = 0.3 * np.sin(3 * t) + 0.1 * complexity * np.sin(s)
            
        elif shape_type == 'shell':
            v_shell = np.linspace(0, np.pi, v_res)
            u_grid, v_grid = np.meshgrid(np.linspace(0, 2*np.pi, u_res), v_shell)
            u, v = u_grid.flatten(), v_grid.flatten()
            a = 0.2 * complexity
            x = a * np.exp(v / (2 * np.pi)) * np.cos(v) * np.cos(u)
            y = a * np.exp(v / (2 * np.pi)) * np.cos(v) * np.sin(u)
            z = a * np.exp(v / (2 * np.pi)) * np.sin(v)
            
        else: # Cube/Organic fallback
            return self.generate_parametric_shape('sphere', resolution)

        # 3. CREATE MESH DATA
        vertices = np.stack([x, y, z], axis=1)
        
        # 4. GENERATE FACES (Corrected stride logic)
        faces = []
        for i in range(v_res - 1):     # Rows
            for j in range(u_res - 1): # Cols
                p1 = i * u_res + j
                p2 = p1 + 1
                p3 = p1 + u_res
                p4 = p3 + 1
                
                # Two triangles per quad
                faces.append([p1, p2, p3])
                faces.append([p2, p4, p3])
                
        return vertices, np.array(faces)
    
    def generate_shape(self, shape_type='sphere', resolution=50, add_detail=True, save_path=None):
        print(f"\nGenerating {shape_type} shape...")
        
        if shape_type in ['sphere', 'torus', 'knot', 'shell']:
            params = self.shape_params.get(shape_type, {'complexity': 1.0})
            vertices, faces = self.generate_parametric_shape(
                shape_type, resolution, params['complexity']
            )
        else:
            if shape_type == 'cube':
                sdf_func = lambda p: self.sdf_box(p, size=[0.8, 0.8, 0.8])
            else:
                sdf_func = lambda p: self.sdf_sphere(p, radius=1.0)
            
            if add_detail:
                base = sdf_func
                sdf_func = lambda p: self.add_noise(p, base(p))
            
            vertices, faces = self.marching_cubes_basic(sdf_func, resolution)
        
        # Create and Process Mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # IMPORTANT: Fix visuals for 3D Viewer
        mesh.fix_normals()
        # Set color to "cornflower blue" so it shows on black backgrounds
        mesh.visual.face_colors = [255, 105, 180, 255] 
        
        if shape_type in ['sphere', 'organic']:
            try:
                trimesh.smoothing.filter_laplacian(mesh, iterations=3)
            except: pass

        if save_path:
            mesh.export(save_path)
            
        return mesh