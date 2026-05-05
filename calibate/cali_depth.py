import cv2
import numpy as np

# ===== ตั้งค่าช่วง depth ที่ valid =====
MIN_DEPTH = 300    # mm (0.3 m)
MAX_DEPTH = 3000   # mm (3.0 m)

depth_frame = None
color_map = None

def mouse_callback(event, x, y, flags, param):
    global depth_frame, color_map

    if event == cv2.EVENT_LBUTTONDOWN:
        if depth_frame is None:
            return

        depth_value = depth_frame[y, x]

        # ตรวจสอบค่าที่ใช้งานได้
        if depth_value == 0 or depth_value < MIN_DEPTH or depth_value > MAX_DEPTH:
            print(f"Point ({x},{y}) : Invalid depth")
        else:
            distance_m = depth_value / 1000.0
            print(f"Point ({x},{y}) : {distance_m:.3f} m")

            # วาดข้อความบนภาพ
            text = f"{distance_m:.2f} m"
            cv2.putText(color_map, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

            cv2.circle(color_map, (x, y), 5, (0, 255, 0), -1)


# ====== ตัวอย่างจำลอง depth (แทนกล้องจริง) ======
# ถ้าใช้ RealSense ให้เอาส่วนนี้ออกแล้วใส่ pipeline แทน
def fake_depth():
    depth = np.random.randint(200, 4000, (480, 640), dtype=np.uint16)
    return depth


cv2.namedWindow("Depth Viewer")
cv2.setMouseCallback("Depth Viewer", mouse_callback)

while True:
    # ====== แทนด้วย depth_frame จากกล้องจริง ======
    depth_frame = fake_depth()

    # normalize + colormap
    depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    color_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # แสดงช่วง valid
    mask = (depth_frame >= MIN_DEPTH) & (depth_frame <= MAX_DEPTH)
    color_map[~mask] = (0, 0, 0)  # จุดที่ไม่ valid = สีดำ

    cv2.imshow("Depth Viewer", color_map)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()