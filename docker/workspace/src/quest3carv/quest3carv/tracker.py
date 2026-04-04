import cv2
import numpy as np
import os

# ==============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ==============================================================================

MAX_POINTS = 2000
MIN_AGE_CONFIDENCE = 8
MAX_PROBATION_FRAMES = 100       # How many frames a point has to reach maturity
REPLENISH_THRESHOLD_RATIO = 0.8
GFTT_QUALITY_LEVEL = 0.10
GFTT_MIN_DISTANCE = 15
MIN_PATCH_VARIANCE = 30.0

KLT_WIN_SIZE = (31, 31)
KLT_MAX_LEVEL = 4

ZNCC_THRESHOLD_STEREO = 0.50
ZNCC_THRESHOLD_TEMPORAL = 0.40
PATCH_SIZE = 15

# Stereo Search Bounds
MAX_DISPARITY = 120              # Minimum physical depth limit (close to headset)
MIN_DISPARITY = 2.5              # Maximum physical depth limit (horizon)
MIN_DEPTH_PROJ = 0.1

VOXEL_SIZE = 0.10                
MIN_SVD_BASELINE = 0.05
SVD_CONDITION_THRESHOLD = 15.0   # Reject degenerate multi-view geometry

# Depth Veto Tolerances
TRUST_DEPTH_AGE = 5              # Age required before enforcing asymmetric depth checks
MAX_OUTWARD_DRIFT = 0.10         # Strict tolerance for points moving away (drills)
MAX_INWARD_DRIFT = -0.30         # Loose tolerance for points moving closer (occlusions)

# Tracking History & Smoothing
EMA_ALPHA = 0.3                  # Weight for Exponential Moving Average smoothing
MAX_HISTORY_FRAMES = 15          # Maximum poses retained for SVD optimization
EPSILON = 1e-6                   # Math safety denominator

# --- THE NEW VETO PARAMETERS ---
EDGE_MARGIN = 40                 # Stay away from cv2.remap black borders!
MAX_DEPTH_ERROR = 0.20           # 20cm tolerance for occlusion/ghost detection

DRAW_DEBUG = True
DEBUG_DIR = "/workspace/debug/tracker_frames"
if DRAW_DEBUG and not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

# ==============================================================================

class StereoPointTracker:
    def __init__(self, K, baseline):
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.baseline = baseline

        self.max_points = MAX_POINTS
        self.min_age_confidence = MIN_AGE_CONFIDENCE
        self.ZNCC_THRESHOLD_TEMPORAL = ZNCC_THRESHOLD_TEMPORAL
        self.ZNCC_THRESHOLD_STEREO = ZNCC_THRESHOLD_STEREO
        self.win_size_klt = KLT_WIN_SIZE
        self.patch_size = PATCH_SIZE
        self.half_p = self.patch_size // 2

        # ACTIVE Tracking Arrays
        self.points_3d = []
        self.points_2d_l = []
        self.birth_patches = None
        self.ages = np.array([], dtype=int)
        self.point_ids = np.array([], dtype=int)

        # Global Map
        self.next_global_id = 0
        self.global_map = {}

        self.prev_gray_l = None
        self.prev_pose = None
        self.frame_idx = 0

    # --------------------------------------------------------------------------
    # CORE PIPELINE
    # --------------------------------------------------------------------------

    def ingest_frame(self, img_l, img_r, current_pose):
        """Main entry point for processing a new stereo pair."""
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        debug_out = img_l.copy() if DRAW_DEBUG else None

        # Initialization
        if self.prev_gray_l is None:
            self.prev_gray_l, self.prev_pose = gray_l, current_pose
            self._replenish(gray_l, gray_r, debug_out, current_pose)
            self._finalize_debug(debug_out)
            return

        # Pipeline Steps
        self._track_temporal(gray_l, current_pose)
        self._verify_appearance(gray_l)
        self._match_stereo_and_update_map(gray_l, gray_r, current_pose, debug_out)
        self._reproject(gray_l, current_pose, debug_out)

        # Replenish if depleted
        if len(self.points_2d_l) < (self.max_points * REPLENISH_THRESHOLD_RATIO):
            self._replenish(gray_l, gray_r, debug_out, current_pose)

        # Clean up the weak features before the next frame ---
        self._cull_probation_failures()

        # State updates
        self.prev_gray_l, self.prev_pose = gray_l, current_pose
        self._finalize_debug(debug_out)

    # --------------------------------------------------------------------------
    # PIPELINE SUB-ROUTINES
    # --------------------------------------------------------------------------

    def _track_temporal(self, gray_l, current_pose):
        """Step 1: Predicts motion using OpenXR pose and refines with KLT."""
        initial_guesses = []
        if len(self.points_3d) > 0:
            tw2c = np.linalg.inv(current_pose)
            R, t = tw2c[:3, :3], tw2c[:3, 3]
            for p_world in self.points_3d:
                p_cam = R @ p_world + t
                if p_cam[2] < -MIN_DEPTH_PROJ:
                    initial_guesses.append([(self.fx * p_cam[0] / -p_cam[2]) + self.cx,
                                            (self.fy * p_cam[1] / -p_cam[2]) + self.cy])
                else:
                    initial_guesses.append([0, 0])
            next_pts_guess = np.array(initial_guesses, dtype=np.float32).reshape(-1, 1, 2)
        else:
            next_pts_guess = np.array(self.points_2d_l, dtype=np.float32).reshape(-1, 1, 2)

        if len(self.points_2d_l) > 0:
            prev_pts_arr = np.array(self.points_2d_l, dtype=np.float32).reshape(-1, 1, 2)
            curr_pts_l, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray_l, gray_l, prev_pts_arr, next_pts_guess,
                winSize=self.win_size_klt, maxLevel=KLT_MAX_LEVEL, flags=cv2.OPTFLOW_USE_INITIAL_FLOW
            )

            if status is not None:
                status = status.reshape(-1).astype(bool)
                self.points_2d_l = curr_pts_l.reshape(-1, 2) 
                self._drop_to_lost(status)

    def _verify_appearance(self, gray_l):
        """Step 2: Vetoes points that have suffered severe perspective distortion."""
        if len(self.points_2d_l) > 0:
            zncc_mask = self._batch_zncc_veto(gray_l, self.points_2d_l)
            self._drop_to_lost(zncc_mask)
            self.ages += 1 # Age increments only if they survive temporal + appearance

    def _epipolar_stereo_search(self, gray_l, gray_r, pts_l):
        H, W = gray_l.shape
        stereo_mask = np.zeros(len(pts_l), dtype=bool)
        disparities = np.zeros(len(pts_l), dtype=np.float32)

        for i, (u, v) in enumerate(pts_l):
            u_i, v_i = int(u), int(v)

            if u_i - self.half_p < 0 or u_i + self.half_p >= W or \
               v_i - self.half_p < 0 or v_i + self.half_p >= H:
                continue

            patch_l = gray_l[v_i - self.half_p : v_i + self.half_p + 1,
                             u_i - self.half_p : u_i + self.half_p + 1]

            # --- THE FIX: Force floats into integers so Numpy slicing doesn't crash ---
            u_min = int(max(self.half_p, u_i - MAX_DISPARITY))
            u_max = int(min(W - self.half_p - 1, u_i - MIN_DISPARITY))

            if u_max <= u_min: 
                continue

            strip_r = gray_r[v_i - self.half_p : v_i + self.half_p + 1,
                             u_min - self.half_p : u_max + self.half_p + 1]

            res = cv2.matchTemplate(strip_r, patch_l, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val >= self.ZNCC_THRESHOLD_STEREO:
                best_u_r = u_min + max_loc[0]

                x = max_loc[0]
                if 0 < x < res.shape[1] - 1:
                    y1, y2, y3 = res[0, x-1], res[0, x], res[0, x+1]
                    denom = (y1 - 2*y2 + y3)
                    if denom != 0:
                        best_u_r += (y1 - y3) / (2 * denom)

                stereo_mask[i] = True
                disparities[i] = u - best_u_r

        return stereo_mask, disparities

    def _match_stereo_and_update_map(self, gray_l, gray_r, current_pose, debug_out):
        """Step 3: Verifies depth, applies EMA smoothing, and triggers SVD on maturity."""
        if len(self.points_2d_l) == 0:
            return

        stereo_mask, disparities = self._epipolar_stereo_search(gray_l, gray_r, self.points_2d_l)

        tw2c = np.linalg.inv(current_pose)
        R_cam, t_cam = tw2c[:3, :3], tw2c[:3, 3]

        for i in range(len(self.points_2d_l)):
            if not stereo_mask[i]: continue

            u_l, v = self.points_2d_l[i]
            disp = disparities[i]
            pid = self.point_ids[i]

            depth = (self.fx * self.baseline) / disp
            local_pt_measured = np.array([(u_l - self.cx) * depth / self.fx, -((v - self.cy) * depth / self.fy), -depth])
            z_measured = -local_pt_measured[2]

            p_cam_expected = R_cam @ self.points_3d[i] + t_cam
            z_expected = -p_cam_expected[2]
            depth_diff = z_measured - z_expected
            
            if self.ages[i] > TRUST_DEPTH_AGE:
                if depth_diff > MAX_OUTWARD_DRIFT: 
                    stereo_mask[i] = False
                    continue
                if depth_diff < MAX_INWARD_DRIFT: 
                    stereo_mask[i] = False
                    continue

            if self.ages[i] <= self.min_age_confidence: 
                world_pt = current_pose[:3, :3] @ local_pt_measured + current_pose[:3, 3]

                smoothed_pt = (1.0 - EMA_ALPHA) * self.points_3d[i] + EMA_ALPHA * world_pt

                self.global_map[pid]['history'].append((current_pose, u_l, v))
                if len(self.global_map[pid]['history']) > MAX_HISTORY_FRAMES:
                    self.global_map[pid]['history'].pop(0)

                if self.ages[i] == self.min_age_confidence:
                    if self._is_history_unique_enough(self.global_map[pid]['history']):
                        optimized_pt = self._triangulate_n_views(self.global_map[pid]['history'])
                        if optimized_pt is not None:
                            local_chk = R_cam @ optimized_pt + t_cam
                            if local_chk[2] < 0: 
                                smoothed_pt = optimized_pt 
                    self.global_map[pid]['history'].clear() 

                self.points_3d[i] = smoothed_pt
                self.global_map[pid]['pt_3d'] = smoothed_pt
                
            self.global_map[pid]['age'] = self.ages[i]

            if DRAW_DEBUG and debug_out is not None:
                cv2.circle(debug_out, (int(u_l), int(v)), 3, (0, 255, 0), -1)

        self._drop_to_lost(stereo_mask)

    # --------------------------------------------------------------------------
    # MAP MANAGEMENT UTILS
    # --------------------------------------------------------------------------
    
    def _is_history_unique_enough(self, history):
        """Checks if the camera has translated enough to provide a good SVD baseline."""
        if len(history) < 2: return False
        
        # Extract the XYZ translation coordinates from all historical poses
        positions = np.array([item[0][:3, 3] for item in history])
        
        # Calculate the distance from the first frame's position to all subsequent frames
        distances = np.linalg.norm(positions - positions[0], axis=1)
        max_dist = np.max(distances)
        
        # If the max distance moved is larger than our required baseline, it's a valid SVD setup
        return max_dist >= MIN_SVD_BASELINE

    def _batch_zncc_veto(self, current_gray, current_pts):
        N = len(current_pts)
        if N == 0: return np.array([], dtype=bool)
        h, w = current_gray.shape
        patches, valid_indices = [], []

        for i, pt in enumerate(current_pts):
            u, v = int(pt[0]), int(pt[1])
            # Use EDGE_MARGIN to avoid cv2.remap black void
            if (u < EDGE_MARGIN or u >= w - EDGE_MARGIN or v < EDGE_MARGIN or v >= h - EDGE_MARGIN):
                continue
            patch = current_gray[v-self.half_p : v+self.half_p+1, u-self.half_p : u+self.half_p+1]
            if patch.shape == (self.patch_size, self.patch_size):
                patches.append(patch)
                valid_indices.append(i)

        full_mask = np.zeros(N, dtype=bool)
        if not patches: return full_mask

        curr_tensor = np.stack(patches).astype(np.float32)
        birth_tensor = self.birth_patches[valid_indices].astype(np.float32)
        
        # --- NEW: NOISE NORMALIZATION VETO ---
        # If the patch drifted into a blurry smudge, ZNCC will mathematically hallucinate a match.
        curr_vars = np.var(curr_tensor, axis=(1, 2))
        
        mean_c = np.mean(curr_tensor, axis=(1, 2), keepdims=True)
        mean_b = np.mean(birth_tensor, axis=(1, 2), keepdims=True)
        c_zero, b_zero = curr_tensor - mean_c, birth_tensor - mean_b
        correlation = np.sum(c_zero * b_zero, axis=(1, 2))
        norm = np.sqrt(np.sum(c_zero**2, axis=(1, 2)) * np.sum(b_zero**2, axis=(1, 2)))
        zncc_scores = correlation / (norm + 1e-6)
        
        # Only pass if ZNCC is good AND it hasn't turned into a featureless smudge
        full_mask[valid_indices] = (zncc_scores > self.ZNCC_THRESHOLD_TEMPORAL) & (curr_vars > MIN_PATCH_VARIANCE * 0.5)
        return full_mask

    def _reproject(self, gray_l, current_pose, debug_out):
        lost_ids = [pid for pid, data in self.global_map.items() if data['state'] == 'LOST']
        if not lost_ids: return

        H, W = gray_l.shape
        pts_3d_lost = np.array([self.global_map[pid]['pt_3d'] for pid in lost_ids])

        tw2c = np.linalg.inv(current_pose)
        R, t = tw2c[:3, :3], tw2c[:3, 3]

        p_cam = (R @ pts_3d_lost.T).T + t
        z_cam = p_cam[:, 2]

        depth_mask = (z_cam < -MIN_DEPTH_PROJ) & (z_cam > -4.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            u = self.fx * (p_cam[:, 0] / -z_cam) + self.cx
            v = self.fy * (p_cam[:, 1] / -z_cam) + self.cy

        bounds_mask = (u >= EDGE_MARGIN) & (u < W - EDGE_MARGIN) & (v >= EDGE_MARGIN) & (v < H - EDGE_MARGIN)
        expected_mask = depth_mask & bounds_mask

        expected_indices = np.where(expected_mask)[0]
        resurrected_ids, res_u, res_v = [], [], []

        for idx in expected_indices:
            pid = lost_ids[idx]
            px_u, px_v = int(u[idx]), int(v[idx])
            # .copy() prevents memory leakage and cross-contamination!
            patch = gray_l[px_v-self.half_p : px_v+self.half_p+1, px_u-self.half_p : px_u+self.half_p+1].copy()

            if patch.shape == (self.patch_size, self.patch_size) and np.var(patch) > MIN_PATCH_VARIANCE * 0.5:
                birth_patch = self.global_map[pid]['patch']
                mean_c, mean_b = np.mean(patch), np.mean(birth_patch)
                c_zero = patch.astype(np.float32) - mean_c
                b_zero = birth_patch.astype(np.float32) - mean_b

                norm = np.sqrt(np.sum(c_zero**2) * np.sum(b_zero**2))
                score = np.sum(c_zero * b_zero) / norm if norm > 1e-6 else 0

                if score > self.ZNCC_THRESHOLD_TEMPORAL:
                    self.global_map[pid]['state'] = 'ACTIVE'
                    resurrected_ids.append(pid)
                    res_u.append(px_u)
                    res_v.append(px_v)
                    if DRAW_DEBUG and debug_out is not None:
                        cv2.circle(debug_out, (px_u, px_v), 6, (255, 255, 0), 2)

        if resurrected_ids:
            self.points_3d.extend([self.global_map[pid]['pt_3d'] for pid in resurrected_ids])
            new_patches = [self.global_map[pid]['patch'] for pid in resurrected_ids]
            new_ages = [self.global_map[pid]['age'] for pid in resurrected_ids]
            new_pts_2d = np.array(list(zip(res_u, res_v)), dtype=np.float32)

            if len(self.points_2d_l) > 0:
                self.points_2d_l = np.vstack((self.points_2d_l, new_pts_2d))
                self.birth_patches = np.vstack((self.birth_patches, new_patches))
                self.ages = np.concatenate((self.ages, new_ages))
                self.point_ids = np.concatenate((self.point_ids, resurrected_ids))
            else:
                self.points_2d_l = new_pts_2d
                self.birth_patches = np.array(new_patches)
                self.ages = np.array(new_ages)
                self.point_ids = np.array(resurrected_ids)

    def _replenish(self, gray_l, gray_r, debug_out, current_pose):
        occupied_voxels = set(self._get_voxel(data['pt_3d']) for data in self.global_map.values())

        # Mask out the black void borders generated by cv2.remap!
        mask = np.zeros_like(gray_l)
        mask[EDGE_MARGIN:-EDGE_MARGIN, EDGE_MARGIN:-EDGE_MARGIN] = 255
        
        for pt in self.points_2d_l:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), GFTT_MIN_DISTANCE, 0, -1)

        new_corners = cv2.goodFeaturesToTrack(
            gray_l, maxCorners=self.max_points-len(self.points_2d_l),
            qualityLevel=GFTT_QUALITY_LEVEL, minDistance=GFTT_MIN_DISTANCE, mask=mask
        )
        if new_corners is None: return

        new_corners = new_corners.reshape(-1, 2)
        
        stereo_mask, disparities = self._epipolar_stereo_search(gray_l, gray_r, new_corners)

        for i in range(len(new_corners)):
            if not stereo_mask[i]: continue

            u_l, v = new_corners[i]
            disp = disparities[i]
                
            # --- BUG FIX: .copy() IS MANDATORY ---
            patch = gray_l[int(v)-self.half_p : int(v)+self.half_p+1, int(u_l)-self.half_p : int(u_l)+self.half_p+1].copy()
            
            if patch.shape == (self.patch_size, self.patch_size) and np.var(patch) >= MIN_PATCH_VARIANCE:

                depth = (self.fx * self.baseline) / disp
                local_pt = np.array([(u_l - self.cx) * depth / self.fx, -((v - self.cy) * depth / self.fy), -depth])
                world_pt = current_pose[:3, :3] @ local_pt + current_pose[:3, 3]

                voxel = self._get_voxel(world_pt)
                if voxel in occupied_voxels: continue
                occupied_voxels.add(voxel)

                pid = self.next_global_id
                self.next_global_id += 1

                self.global_map[pid] = {
                    'pt_3d': world_pt,
                    'patch': patch,
                    'age': 0,
                    'state': 'ACTIVE',
                    'history': [(current_pose, u_l, v)],
                    'birth_frame': self.frame_idx  
                }

                self.points_2d_l = np.vstack([self.points_2d_l, [u_l, v]]) if len(self.points_2d_l) > 0 else np.array([[u_l, v]])
                self.points_3d.append(world_pt)
                self.ages = np.append(self.ages, 0)
                self.point_ids = np.append(self.point_ids, pid)

                if self.birth_patches is None:
                    self.birth_patches = np.array([patch])
                else:
                    self.birth_patches = np.append(self.birth_patches, [patch], axis=0)

                if DRAW_DEBUG and debug_out is not None:
                    cv2.drawMarker(debug_out, (int(u_l), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 6, 1)

    def _cull_probation_failures(self):
        """Removes features that failed to reach maturity within the probation window."""
        dead_pids = []
        for pid, data in self.global_map.items():
            # Only judge points that are currently LOST and haven't reached maturity
            if data['state'] == 'LOST' and data['age'] < self.min_age_confidence:
                time_alive = self.frame_idx - data['birth_frame']
                if time_alive > MAX_PROBATION_FRAMES:
                    dead_pids.append(pid)
        
        # Grim Reaper for failed startups only
        for pid in dead_pids:
            del self.global_map[pid]

    # --------------------------------------------------------------------------
    # MATH UTILS
    # --------------------------------------------------------------------------
    
    def _triangulate_n_views(self, history):
        if len(history) < 2: return None
        
        A = np.zeros((len(history) * 2, 4))
        R_xr_to_cv = np.array([
            [1,  0,  0,  0],
            [0, -1,  0,  0],
            [0,  0, -1,  0],
            [0,  0,  0,  1]
        ])

        for i, (pose, u, v) in enumerate(history):
            T_w_to_c_xr = np.linalg.inv(pose)
            T_w_to_c_cv = R_xr_to_cv @ T_w_to_c_xr
            P = self.K @ T_w_to_c_cv[:3, :4]
            
            A[i*2]     = u * P[2, :] - P[0, :]
            A[i*2 + 1] = v * P[2, :] - P[1, :]
        
        # Capture U, S, and Vt
        U, S, Vt = np.linalg.svd(A)
        
        # --- THE CONDITIONING VETO ---
        # S = [s0, s1, s2, s3]
        if S[3] < 1e-6:
            cond_ratio = float('inf') # Perfect mathematical intersection
        else:
            cond_ratio = S[2] / S[3]

        if cond_ratio < SVD_CONDITION_THRESHOLD:
            return None # The geometry is ambiguous! Reject the optimization.
        # -----------------------------

        X = Vt[-1]
        if abs(X[3]) < 1e-6: return None 
        X = X / X[3]
        
        return X[:3]

    def _get_voxel(self, pt_3d):
        return (int(np.floor(pt_3d[0]/VOXEL_SIZE)),
                int(np.floor(pt_3d[1]/VOXEL_SIZE)),
                int(np.floor(pt_3d[2]/VOXEL_SIZE)))

    def _drop_to_lost(self, mask):
        lost_ids = self.point_ids[~mask]
        for pid in lost_ids:
            if pid in self.global_map:
                self.global_map[pid]['state'] = 'LOST'

        self.points_2d_l = self.points_2d_l[mask]
        self.birth_patches = self.birth_patches[mask]
        self.ages = self.ages[mask]
        self.points_3d = [self.points_3d[i] for i in range(len(mask)) if mask[i]]
        self.point_ids = self.point_ids[mask]

    def get_confident_points(self):
        mask = self.ages >= self.min_age_confidence
        pts_3d = np.array([self.points_3d[i] for i, val in enumerate(mask) if val])
        pts_2d = np.array([self.points_2d_l[i] for i, val in enumerate(mask) if val])
        ages_filt = np.array([self.ages[i] for i, val in enumerate(mask) if val])
        ids_filt = np.array([self.point_ids[i] for i, val in enumerate(mask) if val])
        return pts_3d, pts_2d, ages_filt, ids_filt

    def _finalize_debug(self, debug_out):
        if DRAW_DEBUG and debug_out is not None:
            active_count = len(self.points_2d_l)
            lost_count = sum(1 for data in self.global_map.values() if data['state'] == 'LOST')
            cv2.putText(debug_out, f"F:{self.frame_idx} ACT:{active_count} LST:{lost_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            #cv2.imwrite(os.path.join(DEBUG_DIR, f"frame_latest.jpg"), debug_out)
            cv2.imshow("current_frame", debug_out)
            cv2.waitKey(1)
        self.frame_idx += 1