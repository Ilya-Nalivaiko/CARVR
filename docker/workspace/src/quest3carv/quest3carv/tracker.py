import cv2
import numpy as np
import os

# ==============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ==============================================================================

MAX_POINTS = 500
MIN_AGE_CONFIDENCE = 10
REPLENISH_THRESHOLD_RATIO = 0.8
GFTT_QUALITY_LEVEL = 0.08
GFTT_MIN_DISTANCE = 40
MIN_PATCH_VARIANCE = 50.0

KLT_WIN_SIZE = (31, 31)
KLT_MAX_LEVEL = 4

ZNCC_THRESHOLD = 0.40
PATCH_SIZE = 15

STEREO_DISP_GUESS = 20
MAX_V_DRIFT = 2.0
MIN_DISPARITY = 1.0
MIN_DEPTH_PROJ = 0.1

VOXEL_SIZE = 0.10                # 10cm grid for spatial hashing

# Debugging
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
        self.zncc_threshold = ZNCC_THRESHOLD
        self.win_size_klt = KLT_WIN_SIZE
        self.patch_size = PATCH_SIZE
        self.half_p = self.patch_size // 2

        # ACTIVE Tracking Arrays
        self.points_3d = []
        self.points_2d_l = []
        self.birth_patches = None
        self.ages = np.array([], dtype=int)
        self.point_ids = np.array([], dtype=int)

        # Global Map (No deletion logic for now)
        self.next_global_id = 0
        self.global_map = {}      # dict containing: pt_3d, patch, age, state

        self.prev_gray_l = None
        self.prev_pose = None
        self.frame_idx = 0

    def _get_voxel(self, pt_3d):
        return (int(np.floor(pt_3d[0]/VOXEL_SIZE)),
                int(np.floor(pt_3d[1]/VOXEL_SIZE)),
                int(np.floor(pt_3d[2]/VOXEL_SIZE)))

    def _drop_to_lost(self, mask):
        """Moves points that failed tracking from ACTIVE arrays to LOST state."""
        lost_ids = self.point_ids[~mask]
        for pid in lost_ids:
            if pid in self.global_map:
                self.global_map[pid]['state'] = 'LOST'

        # Filter active arrays
        self.points_2d_l = self.points_2d_l[mask]
        self.birth_patches = self.birth_patches[mask]
        self.ages = self.ages[mask]
        self.points_3d = [self.points_3d[i] for i in range(len(mask)) if mask[i]]
        self.point_ids = self.point_ids[mask]

    def _batch_zncc_veto(self, current_gray, current_pts):
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

        if self.prev_gray_l is None:
            self.prev_gray_l, self.prev_pose = gray_l, current_pose
            self._replenish(gray_l, gray_r, debug_out, current_pose)
            self._finalize_debug(debug_out)
            return

        # 1. Motion Prediction & KLT (Accounting for OpenXR -Z)
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
                self.points_2d_l = curr_pts_l.reshape(-1, 2) # Update 2D positions
                self._drop_to_lost(status)

        # 2. Appearance Verification
        if len(self.points_2d_l) > 0:
            zncc_mask = self._batch_zncc_veto(gray_l, self.points_2d_l)
            self._drop_to_lost(zncc_mask)
            self.ages += 1

        # 3. Stereo Matching & Triangulation
        if len(self.points_2d_l) > 0:
            stereo_guess = self.points_2d_l.copy().astype(np.float32)
            stereo_guess[:, 0] -= STEREO_DISP_GUESS

            res_r, stat_r, _ = cv2.calcOpticalFlowPyrLK(
                gray_l, gray_r, self.points_2d_l.astype(np.float32).reshape(-1, 1, 2),
                stereo_guess.reshape(-1, 1, 2), winSize=self.win_size_klt, maxLevel=KLT_MAX_LEVEL,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW
            )

            if stat_r is not None:
                stat_r = stat_r.reshape(-1).astype(bool)
                res_r = res_r.reshape(-1, 2)

                stereo_mask = np.zeros(len(self.points_2d_l), dtype=bool)
                for i in range(len(self.points_2d_l)):
                    u_l, v = self.points_2d_l[i]
                    u_r = res_r[i][0]
                    v_drift = abs(v - res_r[i][1])

                    if stat_r[i] and v_drift < MAX_V_DRIFT and (u_l - u_r) > MIN_DISPARITY:
                        stereo_mask[i] = True

                        # Triangulate & Smooth
                        local_pt = self._triangulate(u_l, u_r, v)
                        world_pt = current_pose[:3, :3] @ local_pt + current_pose[:3, 3]
                        alpha = max(0.1, 1.0 / (self.ages[i] + 1))
                        smoothed_pt = (1.0 - alpha) * self.points_3d[i] + alpha * world_pt

                        self.points_3d[i] = smoothed_pt

                        pid = self.point_ids[i]
                        self.global_map[pid]['pt_3d'] = smoothed_pt
                        self.global_map[pid]['age'] = self.ages[i]

                        if DRAW_DEBUG:
                            cv2.circle(debug_out, (int(u_l), int(v)), 3, (0, 255, 0), -1)

                self._drop_to_lost(stereo_mask)

        # 4. Map-to-Image Re-projection (Resurrect LOST points)
        self._reproject(gray_l, current_pose, debug_out)

        # 5. Birth New Points
        if len(self.points_2d_l) < (self.max_points * REPLENISH_THRESHOLD_RATIO):
            self._replenish(gray_l, gray_r, debug_out, current_pose)

        self.prev_gray_l, self.prev_pose = gray_l, current_pose
        self._finalize_debug(debug_out)

    def _reproject(self, gray_l, current_pose, debug_out):
        """Projects LOST points into the camera. If found, resurrects them."""
        lost_ids = [pid for pid, data in self.global_map.items() if data['state'] == 'LOST']
        if not lost_ids:
            return

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

        bounds_mask = (u >= self.half_p) & (u < W - self.half_p) & (v >= self.half_p) & (v < H - self.half_p)
        expected_mask = depth_mask & bounds_mask

        expected_indices = np.where(expected_mask)[0]
        resurrected_ids, res_u, res_v = [], [], []

        for idx in expected_indices:
            pid = lost_ids[idx]
            px_u, px_v = int(u[idx]), int(v[idx])
            patch = gray_l[px_v-self.half_p : px_v+self.half_p+1, px_u-self.half_p : px_u+self.half_p+1]

            if patch.shape == (self.patch_size, self.patch_size):
                birth_patch = self.global_map[pid]['patch']
                mean_c, mean_b = np.mean(patch), np.mean(birth_patch)
                c_zero = patch.astype(np.float32) - mean_c
                b_zero = birth_patch.astype(np.float32) - mean_b

                norm = np.sqrt(np.sum(c_zero**2) * np.sum(b_zero**2))
                score = np.sum(c_zero * b_zero) / norm if norm > 1e-6 else 0

                if score > self.zncc_threshold:
                    self.global_map[pid]['state'] = 'ACTIVE'
                    resurrected_ids.append(pid)
                    res_u.append(px_u)
                    res_v.append(px_v)
                    if DRAW_DEBUG and debug_out is not None:
                        cv2.circle(debug_out, (px_u, px_v), 6, (255, 255, 0), 2) # Cyan = Resurrected

        # Re-inject into tracking
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

        mask = np.ones_like(gray_l) * 255
        for pt in self.points_2d_l:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), GFTT_MIN_DISTANCE, 0, -1)

        new_corners = cv2.goodFeaturesToTrack(
            gray_l, maxCorners=self.max_points-len(self.points_2d_l),
            qualityLevel=GFTT_QUALITY_LEVEL, minDistance=GFTT_MIN_DISTANCE, mask=mask
        )
        if new_corners is None: return

        initial_guess = new_corners.copy()
        initial_guess[:, 0, 0] -= STEREO_DISP_GUESS

        res_r, stat_r, _ = cv2.calcOpticalFlowPyrLK(
            gray_l, gray_r, new_corners, initial_guess,
            winSize=self.win_size_klt, maxLevel=KLT_MAX_LEVEL, flags=cv2.OPTFLOW_USE_INITIAL_FLOW
        )

        if stat_r is None: return
        stat_r, res_r, new_corners = stat_r.reshape(-1).astype(bool), res_r.reshape(-1, 2), new_corners.reshape(-1, 2)

        for i in range(len(new_corners)):
            u_l, v = new_corners[i]
            u_r = res_r[i][0]
            if stat_r[i] and (u_l - u_r) > MIN_DISPARITY:
                patch = gray_l[int(v)-self.half_p : int(v)+self.half_p+1, int(u_l)-self.half_p : int(u_l)+self.half_p+1]
                if patch.shape == (self.patch_size, self.patch_size) and np.var(patch) >= MIN_PATCH_VARIANCE:

                    local_pt = self._triangulate(u_l, u_r, v)
                    world_pt = current_pose[:3, :3] @ local_pt + current_pose[:3, 3]

                    voxel = self._get_voxel(world_pt)
                    if voxel in occupied_voxels:
                        continue
                    occupied_voxels.add(voxel)

                    pid = self.next_global_id
                    self.next_global_id += 1

                    self.global_map[pid] = {
                        'pt_3d': world_pt,
                        'patch': patch,
                        'age': 0,
                        'state': 'ACTIVE'
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

    def _triangulate(self, u_l, u_r, v):
        disp = max(MIN_DISPARITY, u_l - u_r)
        depth = (self.fx * self.baseline) / disp
        return np.array([(u_l - self.cx) * depth / self.fx, -((v - self.cy) * depth / self.fy), -depth])

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
            cv2.imwrite(os.path.join(DEBUG_DIR, f"frame_latest.jpg"), debug_out)
        self.frame_idx += 1
