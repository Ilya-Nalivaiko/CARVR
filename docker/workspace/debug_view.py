import cv2
import av
import socket
import struct
import numpy as np
import multiprocessing as mp

# Configuration
WIDTH, HEIGHT = 640, 640
UUID = b"CMPUT428_POSE_ID"
POSE_STRUCT_FMT = "<q7f"

def eye_processor(port, eye_side, frame_queue, pose_dict):
    """Handles network ingestion and decoding for a single eye."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2*1024*1024)
    sock.bind(("0.0.0.0", port))
    
    # Internal FFmpeg threading for the C-decoder
    codec = av.CodecContext.create('h264', 'r')
    codec.thread_type = 'FRAME'
    codec.thread_count = 4

    while True:
        try:
            data, _ = sock.recvfrom(65535)
            
            # 1. Handle Pose SEI
            if len(data) == 60 and data[4] == 0x06 and data[7:23] == UUID:
                pose_bytes = data[23:59]
                pose_dict[eye_side] = struct.unpack(POSE_STRUCT_FMT, pose_bytes)
                continue

            # 2. Decode Video
            packets = codec.parse(data)
            for packet in packets:
                try:
                    frames = codec.decode(packet)
                    for frame in frames:
                        img = frame.to_ndarray(format='bgr24')
                        # Downscale immediately to reduce IPC (Inter-Process Communication) overhead
                        small_img = cv2.resize(img, (WIDTH // 2, HEIGHT // 2))
                        
                        # Only keep the latest frame in queue (non-blocking)
                        if frame_queue.full():
                            try: frame_queue.get_nowait()
                            except: pass
                        frame_queue.put(small_img)
                except av.error.InvalidDataError:
                    continue
        except Exception as e:
            print(f"Error on {eye_side}: {e}")

if __name__ == '__main__':
    # Shared resources
    left_q = mp.Queue(maxsize=1)
    right_q = mp.Queue(maxsize=1)
    manager = mp.Manager()
    poses = manager.dict()

    # Start independent processes
    p_left = mp.Process(target=eye_processor, args=(5000, 'left', left_q, poses))
    p_right = mp.Process(target=eye_processor, args=(5001, 'right', right_q, poses))
    
    p_left.start()
    p_right.start()

    print("Stereo processes active. Press 'q' to quit.")

    try:
        while True:
            # Non-blocking get for smooth rendering
            left_img = left_q.get() if not left_q.empty() else None
            right_img = right_q.get() if not right_q.empty() else None

            if left_img is not None and right_img is not None:
                # Combine eyes into Side-by-Side (SBS)
                sbs = np.hstack([left_img, right_img])
                
                # Draw Pose Info from shared dict
                lp = poses.get('left')
                if lp:
                    ts, px, py, pz = lp[0], lp[1], lp[2], lp[3]
                    cv2.putText(sbs, f"Pose: {px:.2f}, {py:.2f}, {pz:.2f}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("CMPUT 428: Parallel Stereo Decoder", sbs)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        p_left.terminate()
        p_right.terminate()
        cv2.destroyAllWindows()