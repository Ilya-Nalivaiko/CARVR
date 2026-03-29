import cv2
import numpy as np
import os

# ==============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ==============================================================================

# Feature Detection (QUALITY OVER QUANTITY)
MAX_POINTS = 250                 # Sliced in half. Only track the absolute best features.
MIN_AGE_CONFIDENCE = 10          # Require 10 frames of survival before sending to C++.
REPLENISH_THRESHOLD_RATIO = 0.8  # Top off frequently so we always have ~250 good points.
GFTT_QUALITY_LEVEL = 0.08        # 4x Stricter! Reject soft edges; only take sharp, distinct corners.
GFTT_MIN_DISTANCE = 40           # Force points to spread out. No more clumping on a single poster.
MIN_PATCH_VARIANCE = 50.0        # Demand very high texture contrast for new points.

# Optical Flow (KLT) Parameters
KLT_WIN_SIZE = (31, 31)          # Keep the wide search window for fast head movements.
KLT_MAX_LEVEL = 4

# Appearance Verification (ZNCC) Parameters
ZNCC_THRESHOLD = 0.40            # A balanced threshold for perspective distortion.
PATCH_SIZE = 15                  

# Stereo Matching & Depth Constraints
STEREO_DISP_GUESS = 20           
MAX_V_DRIFT = 2.0                # STRICT EPIPOLAR RESTRAINT. If it drifts vertically, it's a false match. Kill it.
MIN_DISPARITY = 1.0              
MIN_DEPTH_PROJ = 0.1             

# Debugging & Logging
SAVE_DEBUG_IMAGES = True         
SHOW_DEBUG_IMAGES = False         
SAVE_DEBUG_STATS = True
DEBUG_DIR = "/workspace/debug/tracker_frames"
STATS_FILENAME = "stats.txt"

# Helper flag so we don't draw unnecessarily
DRAW_DEBUG = SAVE_DEBUG_IMAGES or SHOW_DEBUG_IMAGES

# ==============================================================================

class StereoPointTracker:
    def __init__(self, K, baseline):
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.baseline = baseline
        
        # Load configurable parameters from globals
        self.max_points = MAX_POINTS
        self.min_age_confidence = MIN_AGE_CONFIDENCE
        self.zncc_threshold = ZNCC_THRESHOLD
        self.win_size_klt = KLT_WIN_SIZE
        self.patch_size = PATCH_SIZE
        self.half_p = self.patch_size // 2
        
        # State
        self.points_3d = []      
        self.points_2d_l = []    
        self.birth_patches = None 
        self.ages = np.array([], dtype=int)
        self.prev_gray_l = None
        self.prev_pose = None
        self.point_ids = np.array([], dtype=int)
        self.next_global_id = 0
        
        # Debugging & Logging State
        self.frame_idx = 0
        self.stats_file = os.path.join(DEBUG_DIR, STATS_FILENAME)
        
        if SAVE_DEBUG_IMAGES or SAVE_DEBUG_STATS:
            if not os.path.exists(DEBUG_DIR):
                os.makedirs(DEBUG_DIR)
                
        if SAVE_DEBUG_STATS:
            with open(self.stats_file, 'w') as f:
                f.write("frame,active,births,klt_lost,zncc_lost,stereo_lost\n")

    def _batch_zncc_veto(self, current_gray, current_pts):
        """Vectorized ZNCC check against birth patches."""
        N = len(current_pts)
        if N == 0: return np.array([], dtype=bool)
        h, w = current_gray.shape
        patches, valid_indices = [], []

        for i, pt in enumerate(current_pts):
            u, v = int(pt[0]), int(pt[1])
            if (u < self.half_p or u >= w - self.half_p or v < self.half_p or v >= h - self.half_p):
                continue 
            patch = current_gray[v-self.half_p : v+self.half_p+1, u-self.half_p : u+self.half_p+1]
            if patch.shape == (self.patch_size, self.patch_size):
                patches.append(patch)
                valid_indices.append(i)
        
        full_mask = np.zeros(N, dtype=bool)
        if not patches: return full_mask

        curr_tensor = np.stack(patches).astype(np.float32)
        birth_tensor = self.birth_patches[valid_indices].astype(np.float32)
        mean_c = np.mean(curr_tensor, axis=(1, 2), keepdims=True)
        mean_b = np.mean(birth_tensor, axis=(1, 2), keepdims=True)
        c_zero, b_zero = curr_tensor - mean_c, birth_tensor - mean_b
        correlation = np.sum(c_zero * b_zero, axis=(1, 2))
        norm = np.sqrt(np.sum(c_zero**2, axis=(1, 2)) * np.sum(b_zero**2, axis=(1, 2)))
        zncc_scores = correlation / (norm + 1e-6)
        full_mask[valid_indices] = zncc_scores > self.zncc_threshold
        return full_mask

    def ingest_frame(self, img_l, img_r, current_pose):
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        debug_out = img_l.copy() if DRAW_DEBUG else None
        stats = {'klt_lost': 0, 'zncc_lost': 0, 'stereo_lost': 0, 'births': 0}

        if self.prev_gray_l is None:
            self.prev_gray_l, self.prev_pose = gray_l, current_pose
            stats['births'] = self._replenish(gray_l, gray_r, debug_out, current_pose)
            self._finalize_debug(debug_out, stats)
            return

        # 1. Motion Prediction & KLT
        initial_guesses = []
        if len(self.points_3d) > 0:
            tw2c = np.linalg.inv(current_pose)
            R, t = tw2c[:3, :3], tw2c[:3, 3]
            for p_world in self.points_3d:
                p_cam = R @ p_world + t
                if p_cam[2] > MIN_DEPTH_PROJ:
                    initial_guesses.append([(self.fx * p_cam[0] / p_cam[2]) + self.cx, (self.fy * p_cam[1] / p_cam[2]) + self.cy])
                else: 
                    initial_guesses.append([0, 0])
            next_pts_guess = np.array(initial_guesses, dtype=np.float32).reshape(-1, 1, 2)
        else: 
            next_pts_guess = np.array(self.points_2d_l, dtype=np.float32).reshape(-1, 1, 2)

        prev_pts_arr = np.array(self.points_2d_l, dtype=np.float32).reshape(-1, 1, 2)
        curr_pts_l, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray_l, gray_l, prev_pts_arr, next_pts_guess, 
            winSize=self.win_size_klt, maxLevel=KLT_MAX_LEVEL, flags=cv2.OPTFLOW_USE_INITIAL_FLOW
        )
        
        if status is not None:
            status = status.reshape(-1).astype(bool)
            stats['klt_lost'] = np.sum(~status)
            if DRAW_DEBUG:
                for pt in self.points_2d_l[~status]:
                    cv2.drawMarker(debug_out, tuple(pt.astype(int)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 1)
            
            self.points_2d_l = curr_pts_l.reshape(-1, 2)[status]
            self.birth_patches = self.birth_patches[status]
            self.ages = self.ages[status]
            self.points_3d = [self.points_3d[i] for i in range(len(status)) if status[i]]
            self.point_ids = self.point_ids[status]

        # 2. Appearance Verification
        if len(self.points_2d_l) > 0:
            zncc_mask = self._batch_zncc_veto(gray_l, self.points_2d_l)
            stats['zncc_lost'] = np.sum(~zncc_mask)
            if DRAW_DEBUG:
                for pt in self.points_2d_l[~zncc_mask]:
                    cv2.circle(debug_out, tuple(pt.astype(int)), 5, (0, 165, 255), 1)
            
            self.points_2d_l = self.points_2d_l[zncc_mask]
            self.birth_patches = self.birth_patches[zncc_mask]
            self.ages = self.ages[zncc_mask] + 1
            self.points_3d = [self.points_3d[i] for i in range(len(zncc_mask)) if zncc_mask[i]]
            self.point_ids = self.point_ids[zncc_mask]

        if len(self.points_2d_l) == 0:
            stats['births'] = self._replenish(gray_l, gray_r, debug_out, current_pose)
            self._finalize_debug(debug_out, stats)
            return

        # 3. RANGE-BASED STEREO MATCHING
        stereo_guess = self.points_2d_l.copy().astype(np.float32)
        stereo_guess[:, 0] -= STEREO_DISP_GUESS 
        
        res_r, stat_r, _ = cv2.calcOpticalFlowPyrLK(
            gray_l, gray_r, self.points_2d_l.astype(np.float32).reshape(-1, 1, 2), 
            stereo_guess.reshape(-1, 1, 2), winSize=self.win_size_klt, maxLevel=KLT_MAX_LEVEL, 
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW
        )
        
        if stat_r is not None:
            stat_r, res_r = stat_r.reshape(-1).astype(bool), res_r.reshape(-1, 2)
            final_3d, final_2d, final_patches, final_ages, final_ids = [], [], [], [], []
            for i in range(len(self.points_2d_l)):
                u_l, v = self.points_2d_l[i]
                u_r = res_r[i][0]
                v_drift = abs(v - res_r[i][1])
                
                if stat_r[i] and v_drift < MAX_V_DRIFT and (u_l - u_r) > MIN_DISPARITY:
                    local_pt = self._triangulate(u_l, u_r, v)
                    world_pt = current_pose[:3, :3] @ local_pt + current_pose[:3, 3]
                    
                    # The older the point, the less we trust the new noisy stereo measurement
                    alpha = max(0.1, 1.0 / (self.ages[i] + 1)) 
                    smoothed_pt = (1.0 - alpha) * self.points_3d[i] + alpha * world_pt
                    
                    final_3d.append(smoothed_pt)
                    final_2d.append([u_l, v])
                    final_patches.append(self.birth_patches[i])
                    final_ages.append(self.ages[i])
                    final_ids.append(self.point_ids[i]) # Keep the ID alive
                    if DRAW_DEBUG:
                        cv2.circle(debug_out, (int(u_l), int(v)), 3, (0, 255, 0), -1)
                else:
                    stats['stereo_lost'] += 1
                    if DRAW_DEBUG:
                        cv2.drawMarker(debug_out, tuple(self.points_2d_l[i].astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 6, 1)

            self.points_3d, self.points_2d_l = final_3d, np.array(final_2d) if final_2d else np.empty((0, 2))
            self.birth_patches, self.ages = (np.array(final_patches) if final_patches else None), np.array(final_ages)
            self.point_ids = np.array(final_ids, dtype=int)

        if len(self.points_2d_l) < (self.max_points * REPLENISH_THRESHOLD_RATIO):
            stats['births'] += self._replenish(gray_l, gray_r, debug_out, current_pose)

        self.prev_gray_l, self.prev_pose = gray_l, current_pose
        self._finalize_debug(debug_out, stats)

    def _replenish(self, gray_l, gray_r, debug_out, current_pose):
        """Finds new features using a broad disparity search and anchors them to the world frame."""
        mask = np.ones_like(gray_l) * 255
        for pt in self.points_2d_l: 
            cv2.circle(mask, (int(pt[0]), int(pt[1])), GFTT_MIN_DISTANCE, 0, -1)
            
        new_corners = cv2.goodFeaturesToTrack(
            gray_l, maxCorners=self.max_points-len(self.points_2d_l), 
            qualityLevel=GFTT_QUALITY_LEVEL, minDistance=GFTT_MIN_DISTANCE, mask=mask
        )
        if new_corners is None: return 0

        initial_guess = new_corners.copy()
        initial_guess[:, 0, 0] -= STEREO_DISP_GUESS 
        
        res_r, stat_r, _ = cv2.calcOpticalFlowPyrLK(
            gray_l, gray_r, new_corners, initial_guess, 
            winSize=self.win_size_klt, maxLevel=KLT_MAX_LEVEL, flags=cv2.OPTFLOW_USE_INITIAL_FLOW
        )
        
        if stat_r is None: return 0
        stat_r, res_r, new_corners = stat_r.reshape(-1).astype(bool), res_r.reshape(-1, 2), new_corners.reshape(-1, 2)
        
        birth_count = 0
        for i in range(len(new_corners)):
            u_l, v = new_corners[i]
            u_r = res_r[i][0]
            if stat_r[i] and (u_l - u_r) > MIN_DISPARITY:
                patch = gray_l[int(v)-self.half_p : int(v)+self.half_p+1, int(u_l)-self.half_p : int(u_l)+self.half_p+1]
                if patch.shape == (self.patch_size, self.patch_size) and np.var(patch) >= MIN_PATCH_VARIANCE:
                    
                    local_pt = self._triangulate(u_l, u_r, v)
                    world_pt = current_pose[:3, :3] @ local_pt + current_pose[:3, 3]
                    
                    self.points_2d_l = np.vstack([self.points_2d_l, [u_l, v]]) if len(self.points_2d_l) > 0 else np.array([[u_l, v]])
                    self.points_3d.append(world_pt)
                    self.ages = np.append(self.ages, 0)
                    self.point_ids = np.append(self.point_ids, self.next_global_id)
                    self.next_global_id += 1
                    
                    if self.birth_patches is None: 
                        self.birth_patches = np.array([patch])
                    else: 
                        self.birth_patches = np.append(self.birth_patches, [patch], axis=0)
                        
                    if DRAW_DEBUG and debug_out is not None:
                        cv2.drawMarker(debug_out, (int(u_l), int(v)), (255, 255, 0), cv2.MARKER_DIAMOND, 6, 1)
                    birth_count += 1
        return birth_count

    def _finalize_debug(self, debug_out, stats):
        """Logs metrics to stats.txt and handles live display / saving of debug images."""
        active_count = len(self.points_2d_l)
        
        if DRAW_DEBUG and debug_out is not None:
            cv2.putText(debug_out, f"F:{self.frame_idx} Pts:{active_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if SAVE_DEBUG_IMAGES:
                cv2.imwrite(os.path.join(DEBUG_DIR, f"frame_{self.frame_idx:04d}.jpg"), debug_out)
            
            if SHOW_DEBUG_IMAGES:
                cv2.imshow("Stereo Tracker Live", debug_out)
                cv2.waitKey(1)  # Required to pump OpenCV GUI events
            
        if SAVE_DEBUG_STATS:
            with open(self.stats_file, 'a') as f:
                f.write(f"{self.frame_idx},{active_count},{stats['births']},{stats['klt_lost']},{stats['zncc_lost']},{stats['stereo_lost']}\n")
                
        self.frame_idx += 1

    def _triangulate(self, u_l, u_r, v):
        """Returns the 3D point in the LOCAL camera coordinate frame."""
        disp = max(MIN_DISPARITY, u_l - u_r)
        depth = (self.fx * self.baseline) / disp
        return np.array([(u_l - self.cx) * depth / self.fx, (v - self.cy) * depth / self.fy, depth])

    def get_confident_points(self):
        """Returns points exceeding min_age_confidence."""
        mask = self.ages >= self.min_age_confidence
        pts_3d = np.array([self.points_3d[i] for i, val in enumerate(mask) if val])
        pts_2d = np.array([self.points_2d_l[i] for i, val in enumerate(mask) if val])
        ages_filt = np.array([self.ages[i] for i, val in enumerate(mask) if val])
        ids_filt = np.array([self.point_ids[i] for i, val in enumerate(mask) if val])

        return pts_3d, pts_2d, ages_filt, ids_filt