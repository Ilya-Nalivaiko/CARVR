import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
GRID_SIZE = 64                   # Sparse enough to be clean, dense enough to find lines
MIN_TRANSLATION_METERS = 0.25    # Requires 15cm movement before Bayesian updates begin
MAX_DEPTH_VARIANCE = 0.05   
INITIAL_VARIANCE = 2.0      

# KLT & MVS Config
KLT_WIN = (21, 21)
ZNCC_THRESH = 0.70               # Temporal tracking threshold
ZNCC_THRESH_STEREO = 0.80        # STRICT: User preferred initial seed quality
ZNCC_UNIQUENESS_MARGIN = 0.15
PATCH_SIZE = 20 
HALF_P = PATCH_SIZE // 2
MAX_DISPARITY = 120
MIN_DISPARITY = 2.5

GFTT_QUALITY_LEVEL = 0.20        
REPLENISH_THRESHOLD = 50         

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
    def __init__(self, K, baseline, width=640, height=640):
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

        active_pids = [pid for pid, data in self.global_map.items() if data['state'] in ['ACTIVE', 'MATURE']]
        
        # 1. KLT Temporal Tracking
        if active_pids:
            pts_prev = np.array([[self.global_map[p]['u'], self.global_map[p]['v']] for p in active_pids], dtype=np.float32)
            pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_l, pts_prev, None, winSize=KLT_WIN)
            for i, pid in enumerate(active_pids):
                if status[i][0] and 5 < pts_curr[i][0] < self.W-5 and 5 < pts_curr[i][1] < self.H-5:
                    self.global_map[pid]['u'], self.global_map[pid]['v'] = pts_curr[i][0], pts_curr[i][1]
                    self.global_map[pid]['age'] += 1
                    self.global_map[pid]['history'].append((current_pose, pts_curr[i][0], pts_curr[i][1]))
                    
                    if DRAW_DEBUG:
                        color = (0, 255, 0) if self.global_map[pid]['state'] == 'MATURE' else (255, 255, 0)
                        cv2.circle(debug_out, (int(pts_curr[i][0]), int(pts_curr[i][1])), 3, color, -1)
                else:
                    self.global_map[pid]['state'] = 'LOST'

        # 2. O(1) Spatial Hashing
        occupied_grid = {}
        for pid in [p for p in active_pids if self.global_map[p]['state'] != 'LOST']:
            idx = self._get_grid_idx(self.global_map[pid]['u'], self.global_map[pid]['v'])
            if idx in occupied_grid:
                ex_pid = occupied_grid[idx]
                if self.global_map[ex_pid]['filter'].sigma2 < self.global_map[pid]['filter'].sigma2:
                    self.global_map[pid]['state'] = 'LOST'
                else:
                    self.global_map[ex_pid]['state'] = 'LOST'
                    occupied_grid[idx] = pid
            else:
                occupied_grid[idx] = pid

        # 3. Bayesian Depth Filtering
        for pid, data in self.global_map.items():
            if data['state'] != 'ACTIVE': continue
            
            t_dist = np.linalg.norm(current_pose[:3, 3] - data['birth_pose'][:3, 3])
            
            # Wait for physical baseline to expand before updating filter
            if t_dist >= MIN_TRANSLATION_METERS:
                mu, sigma = data['filter'].mu, np.sqrt(data['filter'].sigma2)
                u_r_min = int(data['u'] - ((self.fx * self.baseline) / max(0.1, mu - 2*sigma)))
                u_r_max = int(data['u'] - ((self.fx * self.baseline) / (mu + 2*sigma)))
                
                score, best_u_r = self._safe_match_template(gray_l, gray_r, data['u'], data['v'], u_r_min, u_r_max)
                
                if score and score > ZNCC_THRESH:
                    m_depth = (self.fx * self.baseline) / max(1.0, data['u'] - best_u_r)
                    data['filter'].update(m_depth, 0.1 / max(0.01, t_dist))
                    
                    depth = data['filter'].mu
                    
                    # OpenXR Coordinates (-Y, -Z)
                    l_pt = np.array([
                        (data['u'] - self.cx) * depth / self.fx, 
                        -((data['v'] - self.cy) * depth / self.fy), 
                        -depth
                    ])
                    
                    data['pt_3d'] = current_pose[:3, :3] @ l_pt + current_pose[:3, 3]
                    
                    if data['filter'].converged:
                        data['state'] = 'MATURE'

        # 4. Replenish
        if sum(1 for d in self.global_map.values() if d['state'] in ['ACTIVE', 'MATURE']) < REPLENISH_THRESHOLD:
            self._replenish(gray_l, gray_r, current_pose, occupied_grid, debug_out)

        self.prev_gray = gray_l
        self._finalize_debug(debug_out)

    def _replenish(self, gray_l, gray_r, pose_w2c, occupied_grid=None, debug_out=None):
        mask = np.ones((self.H, self.W), dtype=np.uint8) * 255
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
                    
                    local_pt = np.array([
                        (u - self.cx) * initial_depth / self.fx, 
                        -((v - self.cy) * initial_depth / self.fy), 
                        -initial_depth
                    ])
                    
                    world_pt = pose_w2c[:3, :3] @ local_pt + pose_w2c[:3, 3]

                    pid = self.next_global_id
                    self.next_global_id += 1
                    self.global_map[pid] = {
                        'u': u, 'v': v, 'birth_pose': pose_w2c, 'age': 0, 'state': 'ACTIVE',
                        'filter': DepthFilter(initial_depth, INITIAL_VARIANCE), 'pt_3d': world_pt, 'history': [(pose_w2c, u, v)]
                    }
                    if DRAW_DEBUG and debug_out is not None:
                        cv2.drawMarker(debug_out, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 6, 1)

    def _finalize_debug(self, debug_out):
        if DRAW_DEBUG and debug_out is not None:
            active_count = sum(1 for data in self.global_map.values() if data['state'] in ['ACTIVE', 'MATURE'])
            mature_count = sum(1 for data in self.global_map.values() if data['state'] == 'MATURE')
            
            label = f"F:{self.frame_idx} ACT:{active_count} MAT:{mature_count}"
            cv2.putText(debug_out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Bayesian LSD Tracker", debug_out)
            cv2.waitKey(1)
        self.frame_idx += 1

    def _point_to_segment_dist(self, p, a, b):
        """Math helper for LSD association."""
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / max(1e-6, np.dot(ab, ab))
        t = max(0.0, min(1.0, t))
        closest = a + t * ab
        return np.linalg.norm(p - closest), t

    def get_confident_points(self):
        """
        Extracts mature points
        """
        pts_3d, p2d, ages, ids = [], [], [], []
        
        # 1. Gather real, physically tracked points
        for pid, data in self.global_map.items():
            if data['state'] == 'MATURE':
                pts_3d.append(data['pt_3d'])
                p2d.append([data['u'], data['v']])
                ages.append(data['age'])
                ids.append(pid)

        return np.array(pts_3d), np.array(p2d), np.array(ages), np.array(ids)