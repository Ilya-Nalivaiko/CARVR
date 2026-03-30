import cv2
import av
import socket
import struct
import numpy as np
import multiprocessing as mp

# --- Configuration ---
CHECKERBOARD = (9, 6)
SQUARE_SIZE_MM = 25.0 # Ensure this is exactly the size of your printed squares in millimeters!
MIN_SAMPLES = 12

WIDTH, HEIGHT = 640, 640
UUID = b"CMPUT428_POSE_ID"
POSE_STRUCT_FMT = "<q7f"

def eye_processor(port, eye_side, frame_queue, pose_dict):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2*1024*1024)
    sock.bind(("0.0.0.0", port))
    
    codec = av.CodecContext.create('h264', 'r')
    codec.thread_type = 'FRAME'
    codec.thread_count = 4

    while True:
        try:
            data, _ = sock.recvfrom(65535)
            if len(data) == 60 and data[4] == 0x06 and data[7:23] == UUID:
                pose_bytes = data[23:59]
                pose_dict[eye_side] = struct.unpack(POSE_STRUCT_FMT, pose_bytes)
                continue

            packets = codec.parse(data)
            for packet in packets:
                try:
                    frames = codec.decode(packet)
                    for frame in frames:
                        img = frame.to_ndarray(format='bgr24') 
                        if frame_queue.full():
                            try: frame_queue.get_nowait()
                            except: pass
                        frame_queue.put(img)
                except av.error.InvalidDataError:
                    continue
        except Exception as e:
            print(f"Error on {eye_side}: {e}")

def run_comprehensive_calibration(objpoints, imgpoints_l, imgpoints_r, shape):
    """Computes Individual, Stereo, and Rectification matrices."""
    print("\n" + "="*50)
    print(" PHASE 1: INDIVIDUAL INTRINSICS")
    print("="*50)
    
    # 1. Individual Calibration (Seeds the stereo calibrator)
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, shape, None, None)
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, shape, None, None)
    
    print(f"Left RMS: {ret_l:.4f} | Right RMS: {ret_r:.4f}")

    print("\n" + "="*50)
    print(" PHASE 2: STEREO EXTRINSICS (Baseline & Toe-in)")
    print("="*50)
    
    # 2. Stereo Calibration
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = cv2.CALIB_FIX_INTRINSIC 
    
    ret_stereo, C1, D1, C2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l, mtx_r, dist_r,
        shape, criteria=criteria, flags=flags
    )
    
    baseline_mm = np.linalg.norm(T)
    baseline_m = baseline_mm / 1000.0
    
    print(f"Stereo RMS Error: {ret_stereo:.4f}")
    print(f"PHYSICAL BASELINE: {baseline_mm:.2f} mm ({baseline_m:.5f} meters)")
    print("ROTATION MATRIX (R) [Toe-in angle]:")
    print(R)

    print("\n" + "="*50)
    print(" PHASE 3: RECTIFICATION MATRICES (Copy to ROS)")
    print("="*50)
    
    # 3. Stereo Rectification (Creates the parallel virtual cameras)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        C1, D1, C2, D2, shape, R, T, alpha=0
    )
    
    print("\n# --- COPY THIS BLOCK INTO spatial_node.py __init__ ---")
    print("self.K_l = np.array(" + np.array2string(C1, separator=', ') + ")")
    print("self.D_l = np.array(" + np.array2string(D1.flatten(), separator=', ') + ")")
    print("self.K_r = np.array(" + np.array2string(C2, separator=', ') + ")")
    print("self.D_r = np.array(" + np.array2string(D2.flatten(), separator=', ') + ")")
    print("")
    print("self.R1 = np.array(" + np.array2string(R1, separator=', ') + ")")
    print("self.R2 = np.array(" + np.array2string(R2, separator=', ') + ")")
    print("self.P1 = np.array(" + np.array2string(P1, separator=', ') + ")")
    print("self.P2 = np.array(" + np.array2string(P2, separator=', ') + ")")
    print(f"self.true_baseline = {baseline_m:.5f}")
    print("# -----------------------------------------------------")


if __name__ == '__main__':
    left_q, right_q = mp.Queue(maxsize=1), mp.Queue(maxsize=1)
    manager = mp.Manager()
    poses = manager.dict()

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints, imgpoints_l, imgpoints_r = [], [], []

    p_left = mp.Process(target=eye_processor, args=(5000, 'left', left_q, poses))
    p_right = mp.Process(target=eye_processor, args=(5001, 'right', right_q, poses))
    p_left.start()
    p_right.start()

    print("\n" + "*"*40)
    print(" COMPREHENSIVE STEREO CALIBRATION")
    print("*"*40)
    print(" [SPACE] - Capture current frame")
    print(" [C]     - Calculate & Print Output")
    print(" [Q]     - Quit")
    print("*"*40)

    try:
        while True:
            left_img = left_q.get() if not left_q.empty() else None
            right_img = right_q.get() if not right_q.empty() else None

            if left_img is not None and right_img is not None:
                key = cv2.waitKey(1) & 0xFF
                
                gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

                if key == ord(' '):
                    ret_l, cor_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
                    ret_r, cor_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)

                    if ret_l and ret_r:
                        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        imgpoints_l.append(cv2.cornerSubPix(gray_l, cor_l, (11,11), (-1,-1), term))
                        imgpoints_r.append(cv2.cornerSubPix(gray_r, cor_r, (11,11), (-1,-1), term))
                        objpoints.append(objp)
                        print(f"Captured sample {len(objpoints)}")
                    else:
                        print("Corners not found in both eyes. Adjust board and try again.")

                elif key == ord('c'):
                    if len(objpoints) >= 12: 
                        h, w = gray_l.shape
                        run_comprehensive_calibration(objpoints, imgpoints_l, imgpoints_r, (w, h))
                    else:
                        print(f"Need at least 12 samples for a good stereo calibration! (Current: {len(objpoints)})")

                elif key == ord('q'):
                    break

                disp_l, disp_r = left_img.copy(), right_img.copy()
                cv2.drawChessboardCorners(disp_l, CHECKERBOARD, None, False)
                cv2.drawChessboardCorners(disp_r, CHECKERBOARD, None, False)
                
                sbs = np.hstack([cv2.resize(disp_l, (400, 400)), 
                                 cv2.resize(disp_r, (400, 400))])
                cv2.putText(sbs, f"Count: {len(objpoints)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Stereo Calibration Matrix", sbs)

    finally:
        p_left.terminate()
        p_right.terminate()
        cv2.destroyAllWindows()