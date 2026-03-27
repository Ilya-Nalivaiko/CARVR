import cv2
import numpy as np

class StereoPointTracker:
    def __init__(self, K, baseline, max_points=500, min_age_confidence=10):
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.baseline = baseline
        self.max_points = max_points
        self.min_age_confidence = min_age_confidence
        
        # Hyperparameters
        self.zncc_threshold = 0.85
        self.patch_size = 15  # Must be odd
        self.half_p = self.patch_size // 2
        
        # State
        self.points_3d = []      
        self.points_2d_l = []    
        self.birth_patches = None # Will be a (N, 15, 15) tensor
        self.ages = np.array([], dtype=int)
        self.prev_gray_l = None
        self.prev_pose = None

    def _batch_zncc_veto(self, current_gray, current_pts):
        """Vectorized ZNCC: Computes all scores in one NumPy operation."""
        N = len(current_pts)
        if N == 0: return np.array([], dtype=bool)

        h, w = current_gray.shape
        patches = []
        valid_indices = []

        for i, pt in enumerate(current_pts):
            u, v = int(pt[0]), int(pt[1])
            
            # --- ADD BOUNDARY CHECK HERE ---
            if (u < self.half_p or u >= w - self.half_p or 
                v < self.half_p or v >= h - self.half_p):
                continue 
                
            patch = current_gray[v-self.half_p : v+self.half_p+1, 
                                u-self.half_p : u+self.half_p+1]
            
            # Double check shape just in case of rounding errors
            if patch.shape == (self.patch_size, self.patch_size):
                patches.append(patch)
                valid_indices.append(i)
        
        # Fixed: return a mask that matches the input dimension N
        full_mask = np.zeros(N, dtype=bool)
        if not patches:
            return full_mask

        # Stack only the valid patches
        curr_tensor = np.stack(patches).astype(np.float32)
        # We must index birth_patches to match the points that survived the boundary check
        birth_tensor = self.birth_patches[valid_indices].astype(np.float32)

        # 2. Vectorized ZNCC math: (A - meanA) * (B - meanB) / (stdA * stdB)
        # Mean across the (15, 15) dimensions
        mean_c = np.mean(curr_tensor, axis=(1, 2), keepdims=True)
        mean_b = np.mean(birth_tensor, axis=(1, 2), keepdims=True)
        
        c_zero = curr_tensor - mean_c
        b_zero = birth_tensor - mean_b
        
        # Dot product across the patch
        correlation = np.sum(c_zero * b_zero, axis=(1, 2))
        
        # Normalization (Variance)
        norm = np.sqrt(np.sum(c_zero**2, axis=(1, 2)) * np.sum(b_zero**2, axis=(1, 2)))
        
        zncc_scores = correlation / (norm + 1e-6)
        full_mask[valid_indices] = zncc_scores > self.zncc_threshold
        return full_mask

    def ingest_frame(self, img_l, img_r, current_pose):
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if self.prev_gray_l is None:
            self.prev_gray_l, self.prev_pose = gray_l, current_pose
            self._replenish(gray_l, gray_r)
            return

        # 1. KLT Temporal Tracking (Frame-to-Frame)
        # We pass self.points_2d_l as the initial guess
        curr_pts_l, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray_l, gray_l, 
            np.array(self.points_2d_l, dtype=np.float32), 
            None, winSize=(21, 21)
        )
        status = status.reshape(-1).astype(bool)

        # 2. The Vectorized Veto
        # Fixed: We apply the KLT status to ALL state variables first to keep them aligned
        self.points_2d_l = curr_pts_l[status]
        self.birth_patches = self.birth_patches[status]
        self.ages = self.ages[status]
        self.points_3d = [self.points_3d[i] for i in range(len(status)) if status[i]]

        if len(self.points_2d_l) == 0:
            self._replenish(gray_l, gray_r)
            return

        # Run the vectorized ZNCC on the points that survived KLT
        zncc_mask = self._batch_zncc_veto(gray_l, self.points_2d_l)
        
        # 3. Final selection of points that survived KLT AND ZNCC
        # Fixed: Update all state variables with zncc_mask
        self.points_2d_l = self.points_2d_l[zncc_mask]
        self.birth_patches = self.birth_patches[zncc_mask]
        self.ages = self.ages[zncc_mask] + 1
        self.points_3d = [self.points_3d[i] for i in range(len(zncc_mask)) if zncc_mask[i]]
        
        # Re-triangulate survived points for fresh 3D data
        # (Optional: In a full SLAM you'd use a filter here)
        new_3d = []
        final_2d = []
        final_patches = []
        final_ages = []

        # 1D Stereo check for depth (Vectorize this via KLT)
        res_r, stat_r, _ = cv2.calcOpticalFlowPyrLK(
            gray_l, gray_r, self.points_2d_l.astype(np.float32), None, winSize=(21, 21)
        )
        stat_r = stat_r.reshape(-1).astype(bool)
        
        for i in range(len(self.points_2d_l)):
            if stat_r[i]:
                u_l, v = self.points_2d_l[i]
                u_r = res_r[i].ravel()[0]
                new_3d.append(self._triangulate(u_l, u_r, v))
                final_2d.append([u_l, v])
                final_patches.append(self.birth_patches[i])
                final_ages.append(self.ages[i])

        self.points_3d = new_3d
        self.points_2d_l = np.array(final_2d) if final_2d else np.empty((0, 2))
        self.birth_patches = np.array(final_patches) if final_patches else None
        self.ages = np.array(final_ages)

        if len(self.points_2d_l) < self.max_points * 0.7:
            self._replenish(gray_l, gray_r)

        self.prev_gray_l, self.prev_pose = gray_l, current_pose

    def _replenish(self, gray_l, gray_r):
        """Adds new high-quality features to the pool."""
        mask = np.ones_like(gray_l) * 255
        for pt in self.points_2d_l:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), 15, 0, -1)

        new_corners = cv2.goodFeaturesToTrack(gray_l, maxCorners=self.max_points-len(self.points_2d_l),
                                             qualityLevel=0.05, minDistance=20, mask=mask)
        if new_corners is None: return

        # Stereo Match the new births
        res_r, stat_r, _ = cv2.calcOpticalFlowPyrLK(gray_l, gray_r, new_corners, None, winSize=(21, 21))
        stat_r = stat_r.reshape(-1).astype(bool)
        
        for i in range(len(new_corners)):
            if stat_r[i]:
                u_l, v = new_corners[i].ravel()
                u_r = res_r[i].ravel()[0]
                
                # Check for "Good Texture" before birth
                patch = gray_l[int(v)-self.half_p : int(v)+self.half_p+1, 
                               int(u_l)-self.half_p : int(u_l)+self.half_p+1]
                if patch.shape != (self.patch_size, self.patch_size): continue
                
                # Texture check (Variance)
                if np.var(patch) < 100: continue 

                self.points_2d_l = np.vstack([self.points_2d_l, [u_l, v]]) if len(self.points_2d_l) > 0 else np.array([[u_l, v]])
                self.points_3d.append(self._triangulate(u_l, u_r, v))
                self.ages = np.append(self.ages, 0)
                
                if self.birth_patches is None: self.birth_patches = np.array([patch])
                else: self.birth_patches = np.append(self.birth_patches, [patch], axis=0)

    def _triangulate(self, u_l, u_r, v):
        disp = max(1.0, u_l - u_r)
        depth = (self.fx * self.baseline) / disp
        return np.array([(u_l - self.cx) * depth / self.fx, (v - self.cy) * depth / self.fy, depth])

    def get_confident_points(self):
        """Returns 3D coords, 2D coords, and ages for points exceeding min_age_confidence."""
        mask = self.ages >= self.min_age_confidence
        conf_3d = [self.points_3d[i] for i, val in enumerate(mask) if val]
        conf_2d = self.points_2d_l[mask]
        conf_ages = self.ages[mask]
        return np.array(conf_3d), conf_2d, conf_ages