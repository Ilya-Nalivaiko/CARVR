import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
GRID_SIZE = 32                   # Sparse enough to be clean, dense enough to find lines
MIN_TRANSLATION_METERS = 0.05    # Don't spam Bayesian updates
MAX_DEPTH_VARIANCE = 0.10   
INITIAL_VARIANCE = 2.0      

# KLT & MVS Config
KLT_WIN = (21, 21)
ZNCC_THRESH_STEREO = 0.90        # STRICT: User preferred initial seed quality
ZNCC_UNIQUENESS_MARGIN = 0.15
PATCH_SIZE = 20 
HALF_P = PATCH_SIZE // 2
MAX_DISPARITY = 120
MIN_DISPARITY = 2.5

GFTT_QUALITY_LEVEL = 0.30        
REPLENISH_THRESHOLD = 50         

# BRISK Config
BRISK_MATCH_LOWE = 0.40          # Hamming distance threshold for a successful match
BRISK_MATCH_CEIL = 70
RECOVERY_SEARCH_WINDOW = 40      # Pixels to search around projected LOST point

# --- DEBUG SETTINGS ---
DRAW_DEBUG = True
DEBUG_DIR = "/workspace/debug/tracker_frames"
if DRAW_DEBUG and not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

class DepthFilter:
    def __init__(self, mu, sigma2):
        self.mu = mu          
        self.sigma2 = sigma2  
        self.converged = False

    def update(self, measured_depth, measurement_variance):
        kalman_gain = self.sigma2 / (self.sigma2 + measurement_variance)
        self.mu = self.mu + kalman_gain * (measured_depth - self.mu)
        self.sigma2 = (1 - kalman_gain) * self.sigma2
        if self.sigma2 < MAX_DEPTH_VARIANCE:
            self.converged = True

class StereoPointTracker:
    def __init__(self, K, baseline, logger, width=640, height=640):
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.baseline = baseline
        self.W, self.H = width, height
        self.grid_cols = self.W // GRID_SIZE
        self.grid_rows = self.H // GRID_SIZE
        self.next_global_id = 0
        self.global_map = {} 
        self.prev_gray = None
        self.frame_idx = 0

        # Initialize BRISK and Matcher
        self.brisk = cv2.BRISK_create(thresh=15)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.logger = logger

    def _get_grid_idx(self, u, v):
        c = int(np.clip(u // GRID_SIZE, 0, self.grid_cols - 1))
        r = int(np.clip(v // GRID_SIZE, 0, self.grid_rows - 1))
        return r * self.grid_cols + c

    def _safe_match_template(self, gray_l, gray_r, u_l, v, u_r_min, u_r_max):
        v_start, v_end = int(int(v) - HALF_P), int(int(v) + HALF_P)
        u_start, u_end = int(int(u_l) - HALF_P), int(int(u_l) + HALF_P)
        
        if v_start < 0 or v_end >= self.H or u_start < 0 or u_end >= self.W:
            return None, 0

        patch_l = gray_l[v_start:v_end, u_start:u_end]
        u_r_start = int(max(0, u_r_min - HALF_P))
        u_r_end = int(min(self.W, u_r_max + HALF_P))

        if (u_r_end - u_r_start) < patch_l.shape[1] or (v_end - v_start) < patch_l.shape[0]:
            return None, 0

        strip_r = gray_r[v_start:v_end, u_r_start:u_r_end]
        if strip_r.shape[0] < patch_l.shape[0] or strip_r.shape[1] < patch_l.shape[1]:
            return None, 0

        res = cv2.matchTemplate(strip_r, patch_l, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        # --- UNIQUENESS CHECK ---
        # Create a copy to find the second peak
        res_copy = res.copy()
        
        # Suppress the area around the best match (e.g., a 5px window)
        # This prevents the "second best" from just being the neighbor of the "best"
        h, w = res.shape
        x, y = max_loc
        r = HALF_P
        y_min, y_max = max(0, y-r), min(h, y+r+1)
        x_min, x_max = max(0, x-r), min(w, x+r+1)
        res_copy[y_min:y_max, x_min:x_max] = -1 # Set to lowest possible ZNCC
        
        # Find the second best peak
        _, second_max_val, _, _ = cv2.minMaxLoc(res_copy)
        
        # Apply the margin check
        # We ensure the best match is significantly better than any other candidate
        if (max_val - second_max_val) < ZNCC_UNIQUENESS_MARGIN:
            return None, 0
        # ----------------------------

        best_u_r = u_r_start + max_loc[0] + HALF_P
        return max_val, best_u_r

    def ingest_frame(self, img_l, img_r, current_pose):
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        debug_out = img_l.copy() if DRAW_DEBUG else None

        if self.prev_gray is None:
            self.prev_gray = gray_l
            self._replenish(gray_l, gray_r, current_pose, debug_out=debug_out)
            self._finalize_debug(debug_out)
            return

        active_pids = [pid for pid, data in self.global_map.items() if data['state'] in ['NEW', 'MATURE']]
        
        # 1. KLT Temporal Tracking
        if active_pids:
            pts_prev = np.array([[self.global_map[p]['u'], self.global_map[p]['v']] for p in active_pids], dtype=np.float32)
            pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_l, pts_prev, None, winSize=KLT_WIN)
            for i, pid in enumerate(active_pids):
                if status[i][0] and (HALF_P+2) < pts_curr[i][0] < self.W-(HALF_P+2) and (HALF_P+2) < pts_curr[i][1] < self.H-(HALF_P+2):
                    self.global_map[pid]['u'], self.global_map[pid]['v'] = pts_curr[i][0], pts_curr[i][1]
                    
                    if DRAW_DEBUG:
                        color = (0, 255, 0) if self.global_map[pid]['state'] == 'MATURE' else (255, 255, 0)
                        cv2.circle(debug_out, (int(pts_curr[i][0]), int(pts_curr[i][1])), 3, color, -1)
                else:
                    # If it was still a seed, don't bother recovering
                    if self.global_map[pid]['state'] == 'NEW':
                        self.global_map[pid]['state'] = 'DEAD'
                        # do not bother recovering points which never matured. deleted later
                    elif self.global_map[pid]['state'] == 'MATURE':
                        # It was MATURE, so mark it LOST for BRISK recovery
                        self.global_map[pid]['state'] = 'LOST'

        # 1.5 BRISK Re-acquisition for LOST points
        lost_pids = [pid for pid, data in self.global_map.items() 
                     if data['state'] == 'LOST' and 'descriptor' in data]
        
        if lost_pids:
            # current_pose is Pose_w2c (World to Camera). Invert it to get Camera to World
            R_w2c = current_pose[:3, :3].T
            t_w2c = -R_w2c @ current_pose[:3, 3]
            
            for pid in lost_pids:
                data = self.global_map[pid]
                pt_w = data['pt_3d']
                
                # Project world point into current camera frame
                pt_c = R_w2c @ pt_w + t_w2c
                depth = -pt_c[2] # Based on your OpenXR coordinate system
                
                if depth < 0.1:
                    #self.logger.warn(f"LOST NOT FOUND: point behind camera")
                    continue # Point is behind the camera
                
                proj_u = (pt_c[0] * self.fx / depth) + self.cx
                proj_v = (-pt_c[1] * self.fy / depth) + self.cy
                
                # Check if projected point is within image bounds + margin
                if not (RECOVERY_SEARCH_WINDOW < proj_u < self.W - RECOVERY_SEARCH_WINDOW and 
                        RECOVERY_SEARCH_WINDOW < proj_v < self.H - RECOVERY_SEARCH_WINDOW):
                    #self.logger.warn(f"LOST NOT FOUND: not in image")
                    continue
                    
                # Create a black mask the size of the whole image
                mask = np.zeros_like(gray_l)

                # Draw a white rectangle over your search window
                u_min, u_max = int(proj_u - RECOVERY_SEARCH_WINDOW), int(proj_u + RECOVERY_SEARCH_WINDOW)
                v_min, v_max = int(proj_v - RECOVERY_SEARCH_WINDOW), int(proj_v + RECOVERY_SEARCH_WINDOW)

                # Clamp coordinates safely to image bounds
                u_min, u_max = max(0, u_min), min(self.W, u_max)
                v_min, v_max = max(0, v_min), min(self.H, v_max)

                mask[v_min:v_max, u_min:u_max] = 255

                # Detect features on the FULL image, but restricted by the mask
                kps = self.brisk.detect(gray_l, mask=mask)
                if not kps:
                    #self.logger.warn(f"LOST NOT FOUND: not kps")
                    continue
                
                kps, descs = self.brisk.compute(gray_l, kps)
                if descs is None:
                    #self.logger.warn(f"LOST NOT FOUND: not descs")
                    continue
                
                # Match against saved descriptor
                matches = self.matcher.knnMatch(np.array([data['descriptor']]), descs, k=2)
                if len(matches[0]) == 2:
                    m, n = matches[0]
                    if m.distance < BRISK_MATCH_LOWE * n.distance and m.distance < BRISK_MATCH_CEIL: # Lowe's Ratio
                        # This is a robust match!
                        best_kp = kps[m.trainIdx]
                    
                        # Recover the point!
                        data['u'] = best_kp.pt[0]
                        data['v'] = best_kp.pt[1]
                        data['state'] = 'MATURE'
                        
                        if DRAW_DEBUG:
                            # Draw recovered points in Magenta
                            cv2.circle(debug_out, (int(data['u']), int(data['v'])), 5, (255, 0, 255), 2)
                            self.logger.warn(f"LOST WAS ACTUALLY FOUND WOW")
                else:
                    #self.logger.warn(f"LOST NOT FOUND: Not mached. Best match distance was {matches[0].distance}")
                    continue

        # 2. O(1) Spatial Hashing
        # need to re-do because of above two steps
        active_pids = [pid for pid, data in self.global_map.items() if data['state'] in ['NEW', 'MATURE']]
        occupied_grid = {}
        for pid in active_pids:
            idx = self._get_grid_idx(self.global_map[pid]['u'], self.global_map[pid]['v'])
            if idx in occupied_grid:
                ex_pid = occupied_grid[idx]
                if self.global_map[ex_pid]['filter'].sigma2 < self.global_map[pid]['filter'].sigma2:
                    if self.global_map[ex_pid]['state'] == 'NEW':
                        self.global_map[ex_pid]['state'] = 'DEAD'
                    elif self.global_map[ex_pid]['state'] == 'MATURE':
                        self.global_map[ex_pid]['state'] = 'LOST'
                else:
                    if self.global_map[ex_pid]['state'] == 'NEW':
                        self.global_map[ex_pid]['state'] = 'DEAD'
                    elif self.global_map[ex_pid]['state'] == 'MATURE':
                        self.global_map[ex_pid]['state'] = 'LOST'
                    occupied_grid[idx] = pid
            else:
                occupied_grid[idx] = pid

        # 3. Bayesian Depth Filtering
        for pid, data in self.global_map.items():
            if data['state'] != 'NEW': continue
            
            # check for baseline change (NOT just translation)
            R_birth_inv = data['birth_pose'][:3, :3].T
            t_world_delta = current_pose[:3, 3] - data['last_update_pose'][:3, 3]
            lateral_dist = np.linalg.norm((R_birth_inv @ t_world_delta)[:2])
            if lateral_dist >= MIN_TRANSLATION_METERS:
                # Perform ZNCC and Kalman Update
                mu, sigma = data['filter'].mu, np.sqrt(data['filter'].sigma2)
                u_r_min = int(data['u'] - ((self.fx * self.baseline) / max(0.1, mu - 2*sigma)))
                u_r_max = int(data['u'] - ((self.fx * self.baseline) / (mu + 2*sigma)))
                
                score, best_u_r = self._safe_match_template(gray_l, gray_r, data['u'], data['v'], u_r_min, u_r_max)
                
                if score and score > ZNCC_THRESH_STEREO:
                    m_depth = (self.fx * self.baseline) / max(1.0, data['u'] - best_u_r)
                    m_variance = (data['filter'].mu**2) / (self.fx * max(0.01, lateral_dist))
                    data['filter'].update(m_depth, m_variance)
                    
                    depth = data['filter'].mu
                    
                    # OpenXR Coordinates (-Y, -Z)
                    l_pt = np.array([
                        (data['u'] - self.cx) * depth / self.fx, 
                        -((data['v'] - self.cy) * depth / self.fy), 
                        -depth
                    ])

                    data['last_update_pose'] = current_pose.copy()
                    
                    if data['filter'].converged:
                        data['state'] = 'MATURE'
                        # delayed triangulation
                        data['pt_3d'] = current_pose[:3, :3] @ l_pt + current_pose[:3, 3]
                        # Compute BRISK descriptor at maturation
                        kp = [cv2.KeyPoint(float(data['u']), float(data['v']), 15)]
                        _, des = self.brisk.compute(gray_l, kp)
                        if des is not None:
                            data['descriptor'] = des[0]

        # 4. Replenish
        if sum(1 for d in self.global_map.values() if d['state'] in ['NEW', 'MATURE']) < REPLENISH_THRESHOLD:
            self._replenish(gray_l, gray_r, current_pose, occupied_grid, debug_out)
        
        # Safe dictionary clean-up
        pids_to_remove = [pid for pid, data in self.global_map.items() if data['state'] == 'DEAD']
        for pid in pids_to_remove:
            del self.global_map[pid]

        self.prev_gray = gray_l
        self._finalize_debug(debug_out)

    def _replenish(self, gray_l, gray_r, pose_w2c, occupied_grid=None, debug_out=None):
        mask = np.ones((self.H, self.W), dtype=np.uint8) * 255
        edge_margin = HALF_P + 5
        mask[:edge_margin, :] = 0
        mask[-edge_margin:, :] = 0
        mask[:, :edge_margin] = 0
        mask[:, -edge_margin:] = 0
        if occupied_grid:
            for idx in occupied_grid.keys():
                r, c = idx // self.grid_cols, idx % self.grid_cols
                mask[r*GRID_SIZE:(r+1)*GRID_SIZE, c*GRID_SIZE:(c+1)*GRID_SIZE] = 0
        
        corners = cv2.goodFeaturesToTrack(gray_l, 100, GFTT_QUALITY_LEVEL, GRID_SIZE, mask=mask)
        
        if corners is not None:
            for pt in corners.reshape(-1, 2):
                u, v = pt[0], pt[1]
                score, best_u_r = self._safe_match_template(gray_l, gray_r, u, v, u-MAX_DISPARITY, u-MIN_DISPARITY)
                
                if score and score > ZNCC_THRESH_STEREO:
                    initial_depth = (self.fx * self.baseline) / max(1.0, u - best_u_r)
                    
                    # DELAY TRIANGULATION
                    # local_pt = np.array([
                    #     (u - self.cx) * initial_depth / self.fx, 
                    #     -((v - self.cy) * initial_depth / self.fy), 
                    #     -initial_depth
                    # ])
                    
                    #world_pt = pose_w2c[:3, :3] @ local_pt + pose_w2c[:3, 3]

                    pid = self.next_global_id
                    self.next_global_id += 1
                    self.global_map[pid] = {
                        'u': u, 'v': v, 'birth_pose': pose_w2c, 'last_update_pose': pose_w2c, 'state': 'NEW',
                        'filter': DepthFilter(initial_depth, INITIAL_VARIANCE), #'pt_3d': world_pt
                    }
                    if DRAW_DEBUG and debug_out is not None:
                        cv2.drawMarker(debug_out, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 6, 1)

    def _finalize_debug(self, debug_out):
        if DRAW_DEBUG and debug_out is not None:
            active_count = sum(1 for data in self.global_map.values() if data['state'] in ['NEW', 'MATURE'])
            mature_count = sum(1 for data in self.global_map.values() if data['state'] == 'MATURE')
            
            label = f"F:{self.frame_idx} ACT:{active_count} MAT:{mature_count}"
            cv2.putText(debug_out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Bayesian LSD Tracker", debug_out)
            cv2.waitKey(1)
        self.frame_idx += 1

    def get_confident_points(self):
        """
        Extracts mature points
        """
        pts_3d, p2d, ids = [], [], []
        
        # 1. Gather real, physically tracked points
        for pid, data in self.global_map.items():
            if data['state'] == 'MATURE':
                pts_3d.append(data['pt_3d'])
                p2d.append([data['u'], data['v']])
                ids.append(pid)

        return np.array(pts_3d), np.array(p2d), np.array(ids)