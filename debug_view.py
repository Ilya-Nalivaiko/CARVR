import cv2
import av
import socket
import struct
import numpy as np

# 1. Setup the UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5000))

# 2. Initialize the Hardware-Accelerated H.264 Decoder
codec = av.CodecContext.create('h264', 'r')

print("Listening on UDP port 5000... Waiting for Quest 3 stream with Pose SEI.")

UUID = b"CMPUT428_POSE_ID"
POSE_STRUCT_FMT = "<q7f" 
pos_text = "Waiting for Position..."
rot_text = "Waiting for Rotation..."

try:
    while True:
        data, addr = sock.recvfrom(65535)
        
        # 3. INTERCEPT THE SEI NAL UNIT
        if len(data) == 60 and data[4] == 0x06 and data[5] == 0x05 and data[7:23] == UUID:
            pose_bytes = data[23:59]
            timestamp, px, py, pz, qx, qy, qz, qw = struct.unpack(POSE_STRUCT_FMT, pose_bytes)
            
            pos_text = f"Pos: X:{px:.3f} Y:{py:.3f} Z:{pz:.3f}"
            rot_text = f"Rot: {qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}"
            continue 
            
        # 4. DECODE THE VIDEO
        packets = codec.parse(data)
        for packet in packets:
            try:
                # Catch the missing dictionary crash here!
                frames = codec.decode(packet)
            except av.error.InvalidDataError:
                # Silently ignore the error until the next Keyframe arrives
                continue
                
            for frame in frames:
                img = frame.to_ndarray(format='bgr24')
                
                # Downscale by 2x
                h, w, _ = img.shape
                img = cv2.resize(img, (w // 2, h // 2))
                h, w, _ = img.shape
                
                # Stamp Position and Rotation
                cv2.putText(img, pos_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, rot_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.drawMarker(img, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                
                cv2.imshow("CMPUT428: Synced Passthrough", img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

except KeyboardInterrupt:
    print("\nClosing stream.")
finally:
    cv2.destroyAllWindows()
    sock.close()
