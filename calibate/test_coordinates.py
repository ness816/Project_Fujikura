import cv2
import numpy as np
from pupil_apriltags import Detector

# =========================
# 1. โหลดไฟล์คาลิเบรต
# =========================
try:
    config = np.load("robot_vision_config_new.npz")
    MTX = config["mtx"]
    DIST = config["dist"]
    NEW_MTX = config["new_mtx"]
    H = config["homography"]
    H_inv = np.linalg.inv(H)
    print("[ระบบ] โหลดค่าคาลิเบรตเรียบร้อย")
except Exception as e:
    print(f"[ข้อผิดพลาด] {e}")
    exit()

# =========================
# OFFSET แยกจตุภาค
# =========================
OFFSET_Q1 = (-7.8, -7.3)  # (90,100) 7 4
OFFSET_Q2 = (6.7, -3.7)  # (-90,100)
OFFSET_Q3 = (+6.5, +4.8)  # (-90,-100)
OFFSET_Q4 = (-11.2, -5.8)  # (90,-100)

CAMERA_HEIGHT_FROM_FLOOR = 410.0
TAG_SIZE = 30.0


# =========================
# เลือก offset ตามจตุภาค
# =========================
def get_quadrant_offset(x, y):
    if x >= 0 and y >= 0:
        return OFFSET_Q1
    elif x < 0 and y >= 0:
        return OFFSET_Q2
    elif x < 0 and y < 0:
        return OFFSET_Q3
    else:
        return OFFSET_Q4


# =========================
# แปลง pixel → world
# =========================
def get_real_world_xy(u, v):
    pixel_point = np.array([u, v, 1], dtype="float32").reshape(3, 1)
    real_point = np.dot(H, pixel_point)
    real_point /= real_point[2]

    x = float(real_point[0])
    y = float(real_point[1])

    off_x, off_y = get_quadrant_offset(x, y)

    corrected_x = x + off_x
    corrected_y = y + off_y
    print(f"RAW: ({x:.1f},{y:.1f}) -> OFFSET: ({corrected_x:.1f},{corrected_y:.1f})")
    return corrected_x, corrected_y


# =========================
# วาด world points (มี offset)
# =========================
def draw_world_points(img, H_inv):
    world_points = [(0, 0), (90, 100), (90, -100), (-90, 100), (-90, -100)]

    for x, y in world_points:
        off_x, off_y = get_quadrant_offset(x, y)

        x_corr = x + off_x
        y_corr = y + off_y

        pt = np.array([x_corr, y_corr, 1], dtype="float32").reshape(3, 1)
        pix = np.dot(H_inv, pt)
        pix /= pix[2]

        px, py = int(pix[0]), int(pix[1])

        cv2.circle(img, (px, py), 6, (0, 255, 255), -1)

        # ✅ FIX putText (ไม่ error แล้ว)
        """cv2.putText(
            img, f"({x},{y})", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
        )"""


# =========================
# AprilTag setup
# =========================
half_s = TAG_SIZE / 2
object_points = np.array(
    [[-half_s, half_s, 0], [half_s, half_s, 0], [half_s, -half_s, 0], [-half_s, -half_s, 0]],
    dtype=np.float32,
)

at_detector = Detector(families="tag36h11")
cap = cv2.VideoCapture(0)

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    raw_frame = frame.copy()
    undistorted_img = cv2.undistort(frame, MTX, DIST, None, NEW_MTX)

    gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    results = at_detector.detect(gray)

    for det in results:
        u, v = det.center
        real_x, real_y = get_real_world_xy(u, v)

        image_points = np.array(det.corners, dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, NEW_MTX, None)

        if success:
            dist_to_cam = float(np.linalg.norm(tvec))
            obj_height = CAMERA_HEIGHT_FROM_FLOOR - float(tvec[2])

            cx, cy = int(u), int(v)

            for img in [undistorted_img, raw_frame]:
                cv2.circle(img, (cx, cy), 7, (0, 255, 0), -1)

                for c in det.corners:
                    cv2.circle(img, tuple(c.astype(int)), 4, (255, 0, 0), -1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, f"X:{real_x:.1f}", (cx + 25, cy - 20), font, 0.5, (0, 0, 255), 2)
                cv2.putText(img, f"Y:{real_y:.1f}", (cx + 25, cy), font, 0.5, (0, 0, 255), 2)
                cv2.putText(
                    img, f"H:{obj_height:.1f}", (cx + 25, cy + 20), font, 0.5, (0, 0, 255), 2
                )
                cv2.putText(
                    img, f"D:{dist_to_cam:.1f}", (cx + 25, cy + 40), font, 0.5, (0, 0, 255), 2
                )

    # วาด grid world
    draw_world_points(undistorted_img, H_inv)
    draw_world_points(raw_frame, H_inv)

    cv2.imshow("Undistorted View", undistorted_img)
    # cv2.imshow("Raw View (No Calibration)", raw_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
