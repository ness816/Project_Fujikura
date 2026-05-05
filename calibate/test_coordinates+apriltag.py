import cv2
import numpy as np
import pyrealsense2 as rs

# ===== โหลด calibration =====
try:
    config = np.load("robot_vision_config.npz")
    MTX, DIST, NEW_MTX, H = config["mtx"], config["dist"], config["new_mtx"], config["homography"]
    H_inv = np.linalg.inv(H)
    print("[ระบบ] โหลดค่าคาลิเบรตเรียบร้อย")
except:
    print("[ข้อผิดพลาด] ไม่พบไฟล์ robot_vision_config.npz")
    exit()

OFFSET_X = 5.475
OFFSET_Y = 4.15

# ===== Depth range =====
MIN_DEPTH = 300   # mm
MAX_DEPTH = 3000  # mm

# ===== RealSense setup =====
pipeline = rs.pipeline()
config_rs = rs.config()
config_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config_rs)

depth_frame = None
color_map = None

# ===== ฟังก์ชันแปลงพิกัด =====
def get_real_world_xy(u, v):
    pixel_point = np.array([u, v, 1], dtype="float32").reshape(3, 1)
    real_point = np.dot(H, pixel_point)
    real_point /= real_point[2]

    corrected_x = float(real_point[0]) - OFFSET_X
    corrected_y = float(real_point[1]) - OFFSET_Y
    return corrected_x, corrected_y

# ===== Mouse callback =====
def mouse_callback(event, x, y, flags, param):
    global depth_frame, color_map

    if event == cv2.EVENT_LBUTTONDOWN:
        if depth_frame is None:
            return

        depth_value = depth_frame[y, x]

        if depth_value == 0 or depth_value < MIN_DEPTH or depth_value > MAX_DEPTH:
            print(f"[{x},{y}] depth ใช้ไม่ได้")
            return

        # ===== ระยะจากกล้อง =====
        distance_m = depth_value / 1000.0

        # ===== world coordinate =====
        world_x, world_y = get_real_world_xy(x, y)

        print(f"Pixel ({x},{y})")
        print(f"→ Distance: {distance_m:.3f} m")
        print(f"→ World: X={world_x:.2f} mm, Y={world_y:.2f} mm")

        # ===== แสดงบนภาพ =====
        text1 = f"{distance_m:.2f} m"
        text2 = f"({world_x:.0f},{world_y:.0f}) mm"

        cv2.circle(color_map, (x, y), 5, (0,255,0), -1)
        cv2.putText(color_map, text1, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(color_map, text2, (x, y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

# ===== Window =====
cv2.namedWindow("Depth Viewer")
cv2.setMouseCallback("Depth Viewer", mouse_callback)

# ===== Loop =====
while True:
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()

    if not depth or not color:
        continue

    depth_frame = np.asanyarray(depth.get_data())
    color_image = np.asanyarray(color.get_data())

    # ===== Undistort (ใช้ calibration ของคุณ) =====
    color_image = cv2.undistort(color_image, MTX, DIST, None, NEW_MTX)

    # ===== ทำ colormap =====
    depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    color_map = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # ===== mask valid range =====
    mask = (depth_frame >= MIN_DEPTH) & (depth_frame <= MAX_DEPTH)
    color_map[~mask] = (0, 0, 0)

    cv2.imshow("Depth Viewer", color_map)

    if cv2.waitKey(1) & 0xFF == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()