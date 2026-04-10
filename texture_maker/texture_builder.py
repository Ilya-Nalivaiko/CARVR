import os
import json
import cv2
import numpy as np
import trimesh
import math
import gco

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET_DIR = "../docker/workspace/export_dataset"
KF_DIR = os.path.join(DATASET_DIR, "keyframes")
MESH_PATH = os.path.join(DATASET_DIR, "final_mesh.obj")
OUTPUT_DIR = "./textured_output"

# The Rectified Camera Intrinsics (Must match your SLAM P1 matrix!)
FX, FY = 440.9, 440.9
CX, CY = 320.3, 320.6
IMG_W, IMG_H = 640, 640

# Atlas Settings
FACE_RES = 32  # Every triangle gets a 32x32 pixel block in the atlas
# ==============================================================================

def load_dataset():
    print("[1/5] Loading Mesh and Keyframes...")
    mesh = trimesh.load(MESH_PATH, process=False)
    
    keyframes = []
    for filename in sorted(os.listdir(KF_DIR)):
        if filename.endswith(".json"):
            base_name = filename[:-5]
            json_path = os.path.join(KF_DIR, filename)
            img_path = os.path.join(KF_DIR, f"{base_name}.png")
            
            if os.path.exists(img_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                pose = data['pose']
                p = pose['position']
                q = pose['orientation']
                
                from scipy.spatial.transform import Rotation as R
                rot = R.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
                
                T_w2c = np.eye(4)
                T_w2c[:3, :3] = rot.T
                T_w2c[:3, 3] = -rot.T @ np.array([p['x'], p['y'], p['z']])
                
                # --- THE FIX: OpenXR to OpenCV Optical Frame Flip ---
                # OpenXR looks down -Z with +Y Up. OpenCV looks down +Z with +Y Down.
                # This matrix perfectly aligns the camera so the math works!
                R_flip = np.array([
                    [ 1,  0,  0,  0],
                    [ 0, -1,  0,  0],
                    [ 0,  0, -1,  0],
                    [ 0,  0,  0,  1]
                ])
                T_w2c = R_flip @ T_w2c
                # ----------------------------------------------------
                
                img = cv2.imread(img_path)
                
                keyframes.append({
                    'id': data['frame_id'],
                    'T_w2c': T_w2c,
                    'cam_pos': np.array([p['x'], p['y'], p['z']]),
                    'image': img
                })
                
    print(f"      Loaded {len(mesh.faces)} faces and {len(keyframes)} keyframes.")
    return mesh, keyframes

def assign_faces_to_cameras_simple(mesh, keyframes):
    print("[2/5] Calculating Optimal Views & Raycasting Occlusions...")
    face_assignments = {} # face_index -> kf_index
    
    centroids = mesh.triangles_center
    normals = mesh.face_normals
    
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    
    # --- DIAGNOSTIC COUNTERS ---
    stats = {
        'total_evals': 0, 'fail_dist': 0, 'fail_angle': 0, 
        'fail_frustum_z': 0, 'fail_frustum_uv': 0, 'fail_occlusion': 0
    }
    
    for f_idx in range(len(mesh.faces)):
        best_score = -1.0
        best_kf = -1
        
        c = centroids[f_idx]
        n = normals[f_idx]
        
        for kf_idx, kf in enumerate(keyframes):
            stats['total_evals'] += 1
            cam_pos = kf['cam_pos']
            
            view_vec = cam_pos - c
            dist = np.linalg.norm(view_vec)
            if dist < 0.1 or dist > 4.0: 
                stats['fail_dist'] += 1
                continue 
            
            view_vec /= dist
            
            # FIX 1: Use abs() because Delaunay normals might be flipped backwards!
            score = abs(np.dot(n, view_vec))
            if score < 0.2: 
                stats['fail_angle'] += 1
                continue 
            
            p_cam = kf['T_w2c'][:3, :3] @ c + kf['T_w2c'][:3, 3]
            if p_cam[2] <= 0: 
                stats['fail_frustum_z'] += 1
                continue
            
            u = (FX * p_cam[0] / p_cam[2]) + CX
            v = (FY * p_cam[1] / p_cam[2]) + CY
            if not (10 <= u < IMG_W-10 and 10 <= v < IMG_H-10): 
                stats['fail_frustum_uv'] += 1
                continue
            
            # FIX 2: Distance-based occlusion check (Ignores adjacent edge hits)
            ray_origins = np.array([cam_pos])
            ray_dirs = np.array([-view_vec])
            locations, index_ray, index_tri = intersector.intersects_location(ray_origins, ray_dirs, multiple_hits=False)
            
            if len(locations) > 0:
                hit_dist = np.linalg.norm(locations[0] - cam_pos)
                # If the ray hit something more than 5cm closer than our target face, it's a real wall!
                if hit_dist < (dist - 0.05):
                    stats['fail_occlusion'] += 1
                    continue
                
            if score > best_score:
                best_score = score
                best_kf = kf_idx
                
        if best_kf != -1:
            face_assignments[f_idx] = best_kf
            
    # --- PRINT DIAGNOSTICS ---
    print("\n      --- DIAGNOSTIC RESULTS ---")
    print(f"      Total Camera-Face Pairs Checked: {stats['total_evals']}")
    print(f"      Failed Distance (<0.1m or >4.0m) : {stats['fail_dist']}")
    print(f"      Failed Angle (Edge-on)           : {stats['fail_angle']}")
    print(f"      Failed Frustum (Behind Camera)   : {stats['fail_frustum_z']}")
    print(f"      Failed Frustum (Off Screen)      : {stats['fail_frustum_uv']}")
    print(f"      Failed Occlusion (Hit a Wall)    : {stats['fail_occlusion']}")
    print("      --------------------------\n")
            
    print(f"      Successfully assigned {len(face_assignments)} / {len(mesh.faces)} faces to cameras.")
    return face_assignments

def assign_faces_to_cameras_graph_cut(mesh, keyframes):
    print("[2/5] Calculating Optimal Views (Graph Cut Optimization)...")
    
    num_faces = len(mesh.faces)
    num_cams = len(keyframes)
    
    if num_cams == 0:
        return {}

    # --- COST WEIGHTS ---
    # Since we want to prioritize continuity over angle:
    W_SMOOTH = 10.0   # High penalty for changing cameras across an edge
    W_DATA = 1.0      # Lower penalty for sub-optimal viewing angles
    
    # GCO requires integer math. We scale our float costs up by 1000.
    COST_MULT = 1000 
    INVALID_COST = 100000000  # Cost for faces a camera literally cannot see
    
    # 1. Initialize the Data Cost Matrix (Faces x Cameras)
    data_cost = np.full((num_faces, num_cams), INVALID_COST, dtype=np.int32)
    
    centroids = mesh.triangles_center
    normals = mesh.face_normals
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    
    # Calculate Data Costs (E_data)
    for kf_idx, kf in enumerate(keyframes):
        cam_pos = kf['cam_pos']
        T_w2c = kf['T_w2c']
        
        # Vectorized calculations for all faces relative to this camera
        view_vecs = cam_pos - centroids
        dists = np.linalg.norm(view_vecs, axis=1)
        
        # Avoid division by zero
        valid_dists_mask = (dists >= 0.1) & (dists <= 4.0)
        view_vecs[valid_dists_mask] /= dists[valid_dists_mask, np.newaxis]
        
        # Calculate dot products (angles)
        scores = np.abs(np.einsum('ij,ij->i', normals, view_vecs))
        
        # Project all centroids into the camera
        pts_cam = (T_w2c[:3, :3] @ centroids.T).T + T_w2c[:3, 3]
        
        # Identify valid faces before running the expensive raycaster
        for f_idx in range(num_faces):
            if not valid_dists_mask[f_idx]: continue
            if scores[f_idx] < 0.2: continue
            
            p_cam = pts_cam[f_idx]
            if p_cam[2] <= 0: continue
            
            u = (FX * p_cam[0] / p_cam[2]) + CX
            v = (FY * p_cam[1] / p_cam[2]) + CY
            if not (10 <= u < IMG_W-10 and 10 <= v < IMG_H-10): continue
            
            # Distance-based occlusion check
            ray_origins = np.array([cam_pos])
            ray_dirs = np.array([-view_vecs[f_idx]])
            locations, _, _ = intersector.intersects_location(ray_origins, ray_dirs, multiple_hits=False)
            
            if len(locations) > 0:
                hit_dist = np.linalg.norm(locations[0] - cam_pos)
                if hit_dist < (dists[f_idx] - 0.05):
                    continue # Occluded
            
            # --- CALCULATE VALID COST ---
            # Perfect angle (score=1.0) -> cost 0
            # Terrible angle (score=0.2) -> cost 0.8
            angle_cost = 1.0 - scores[f_idx]
            final_cost = int((angle_cost * W_DATA) * COST_MULT)
            data_cost[f_idx, kf_idx] = final_cost

    # 2. Setup the Smoothness Cost (E_smooth)
    # This acts as a "Potts model" - 0 cost if cameras match, W_SMOOTH if they differ
    smooth_cost = np.full((num_cams, num_cams), int(W_SMOOTH * COST_MULT), dtype=np.int32)
    np.fill_diagonal(smooth_cost, 0)
    
    # 3. Setup the Graph Edges (Adjacency)
    # trimesh gives us an (N, 2) array of adjacent face indices, which is exactly what GCO wants
    if len(mesh.face_adjacency) == 0:
        print("      [!] No shared edges detected (polygon soup). Welding vertices...")
        mesh.merge_vertices()
        
    edges = mesh.face_adjacency.astype(np.int32)
    
    # Final safety net just in case the mesh is literally a cloud of floating triangles
    if len(edges) == 0:
        raise ValueError("Mesh has zero connected edges even after welding! Graph Cut requires a connected surface.")
        
    edge_weights = np.ones(len(edges), dtype=np.int32) # Uniform weight for all edges
    
    # 4. Run the Graph Cut Algorithm
    print("      Running Alpha-Expansion Graph Cut...")
    labels = gco.cut_general_graph(edges, edge_weights, data_cost, smooth_cost)

    # 5. Extract Valid Assignments
    face_assignments = {}
    for f_idx in range(num_faces):
        best_kf = labels[f_idx]
        # Check if the solver had to assign a camera that can't actually see the face
        if data_cost[f_idx, best_kf] < INVALID_COST:
            face_assignments[f_idx] = best_kf
            
    print(f"      Successfully assigned {len(face_assignments)} / {len(mesh.faces)} faces to contiguous patches.")
    return face_assignments

def build_texture_atlas(mesh, keyframes, assignments):
    print("[3/5] UV Unwrapping and Extracting Textures...")
    
    num_assigned_faces = len(assignments)
    if num_assigned_faces == 0:
        raise ValueError("No faces were visible to any camera!")
        
    # Calculate Atlas Grid Size
    faces_per_row = math.ceil(math.sqrt(num_assigned_faces))
    atlas_size = faces_per_row * FACE_RES
    
    # Create blank atlas image
    atlas_img = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    
    # Array to hold the new 2D UV coordinates for the OBJ file
    # Format: [u1, v1, u2, v2, u3, v3] for each assigned face
    face_uvs = {} 
    
    current_face_count = 0
    
    for f_idx, kf_idx in assignments.items():
        kf = keyframes[kf_idx]
        T_w2c = kf['T_w2c']
        img = kf['image']
        
        # 1. Project the 3 3D vertices into the 2D keyframe image
        vert_indices = mesh.faces[f_idx]
        pts_3d = mesh.vertices[vert_indices]
        
        pts_cam = (T_w2c[:3, :3] @ pts_3d.T).T + T_w2c[:3, 3]
        
        u = (FX * pts_cam[:, 0] / pts_cam[:, 2]) + CX
        v = (FY * pts_cam[:, 1] / pts_cam[:, 2]) + CY
        src_pts = np.column_stack((u, v)).astype(np.float32)
        
        # 2. Determine where this face lives in the giant Atlas Grid
        row = current_face_count // faces_per_row
        col = current_face_count % faces_per_row
        
        px_x = col * FACE_RES
        px_y = row * FACE_RES
        
        # We map the triangle to a fixed right-triangle in the designated 32x32 block
        # pt1 = Bottom Left, pt2 = Bottom Right, pt3 = Top Left
        dst_pts = np.array([
            [px_x, px_y + FACE_RES - 1], 
            [px_x + FACE_RES - 1, px_y + FACE_RES - 1], 
            [px_x, px_y]
        ], dtype=np.float32)
        
        # 3. Cut and Warp! (Extract the texture patch)
        M = cv2.getAffineTransform(src_pts, dst_pts)
        warped_patch = cv2.warpAffine(img, M, (atlas_size, atlas_size), borderMode=cv2.BORDER_REPLICATE)
        
        # Mask out just the triangle we warped and copy it to the atlas
        mask = np.zeros((atlas_size, atlas_size), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
        atlas_img = np.where(mask[:, :, None] == 255, warped_patch, atlas_img)
        
        # 4. Record Normalized UV Coordinates (0.0 to 1.0) for the OBJ file
        uvs = dst_pts / atlas_size
        # OBJ files expect V=0 at the bottom, OpenCV has V=0 at the top. Flip V!
        uvs[:, 1] = 1.0 - uvs[:, 1] 
        face_uvs[f_idx] = uvs
        
        current_face_count += 1
        
    return atlas_img, face_uvs

def export_textured_obj(mesh, face_uvs):
    print("[4/5] Writing Output Files...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    obj_path = os.path.join(OUTPUT_DIR, "textured_scan.obj")
    mtl_path = os.path.join(OUTPUT_DIR, "material.mtl")
    
    # 1. Write the Material File
    with open(mtl_path, "w") as f:
        f.write("newmtl scan_mat\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("map_Kd atlas.png\n")

    # 2. Write the OBJ File
    with open(obj_path, "w") as f:
        f.write("mtllib material.mtl\n")
        f.write("usemtl scan_mat\n\n")
        
        # Write Vertices
        for v in mesh.vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            
        f.write("\n")
        
        # Write UVs and Faces
        uv_counter = 1
        for f_idx in range(len(mesh.faces)):
            if f_idx in face_uvs:
                # Write the 3 UV coordinates for this face
                uv_data = face_uvs[f_idx]
                f.write(f"vt {uv_data[0][0]:.5f} {uv_data[0][1]:.5f}\n")
                f.write(f"vt {uv_data[1][0]:.5f} {uv_data[1][1]:.5f}\n")
                f.write(f"vt {uv_data[2][0]:.5f} {uv_data[2][1]:.5f}\n")
                
                # Write the Face (VertexID/UV_ID VertexID/UV_ID VertexID/UV_ID)
                v_ids = mesh.faces[f_idx] + 1 # OBJ is 1-indexed
                f.write(f"f {v_ids[0]}/{uv_counter} {v_ids[1]}/{uv_counter+1} {v_ids[2]}/{uv_counter+2}\n")
                
                uv_counter += 3
            else:
                # Untextured faces just get vertex data
                v_ids = mesh.faces[f_idx] + 1
                f.write(f"f {v_ids[0]} {v_ids[1]} {v_ids[2]}\n")

    print(f"      Saved to {OUTPUT_DIR}/textured_scan.obj")

if __name__ == "__main__":
    mesh, keyframes = load_dataset()
    assignments = assign_faces_to_cameras_graph_cut(mesh, keyframes)
    
    atlas_img, face_uvs = build_texture_atlas(mesh, keyframes, assignments)
    
    export_textured_obj(mesh, face_uvs)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "atlas.png"), atlas_img)
    
    print("[5/5] Pipeline Complete! Load 'textured_scan.obj' into Blender.")