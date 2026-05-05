import cv2
import numpy as np
from pupil_apriltags import Detector

# =========================
# โหลด calibration
# =========================
config = np.load("robot_vision_config_clean_new.npz")
MTX = config["mtx"]
DIST = config["dist"]
NEW_MTX = config["new_mtx"]
H = config["homography"]
H_inv = np.linalg.inv(H)


# =========================
# Homography: pixel → world
# =========================
def get_real_world_xy(u, v):
    pixel_point = np.array([u, v, 1], dtype="float32").reshape(3, 1)
    real_point = np.dot(H, pixel_point)
    real_point /= real_point[2]

    x = float(real_point[0])
    y = float(real_point[1])

    return x, y


# =========================
# วาด grid world พร้อมแสดงพิกัด (x, y)
# =========================
# =========================
# วาดพิกัดโลกโดยเริ่มนับจากกึ่งกลางจอ (0,0)
# =========================
def draw_world_points(img):
    font = cv2.FONT_HERSHEY_SIMPLEX

    h_img, w_img = img.shape[:2]

    range_val_y = 100  # ขอบเขต (mm)
    range_val_x = 90  # ขอบเขต (mm)

    for x in range(-range_val_x, range_val_x + 1, 10):
        for y in range(-range_val_y, range_val_y + 1, 20):
            # ใช้ world จริงเลย (ไม่ offset)
            pt = np.array([x, y, 1], dtype="float32").reshape(3, 1)
            pix = np.dot(H_inv, pt)
            pix /= pix[2]

            px, py = int(pix[0]), int(pix[1])

            if 0 <= px < w_img and 0 <= py < h_img:
                if x == 0 and y == 0:
                    # origin จริง
                    cv2.circle(img, (px, py), 2, (0, 0, 255), -1)
                    cv2.putText(img, "(0,0)", (px + 5, py - 5), font, 0.6, (0, 0, 255), 2)
                else:
                    cv2.circle(img, (px, py), 2, (255, 255, 255), -1)


def draw_boundary(img):
    # เอาเฉพาะ 4 มุม (ไม่เอา (0,0))
    world_points = [(90, 100), (90, -100), (-90, -100), (-90, 100)]

    pixel_points = []

    for x, y in world_points:
        pt = np.array([x, y, 1], dtype="float32").reshape(3, 1)
        pix = np.dot(H_inv, pt)
        pix /= pix[2]

        px, py = int(pix[0]), int(pix[1])
        pixel_points.append((px, py))

    # วาดเส้นเชื่อมเป็นกรอบ
    for i in range(len(pixel_points)):
        pt1 = pixel_points[i]
        pt2 = pixel_points[(i + 1) % len(pixel_points)]  # ปิด loop
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)


# =========================
# AprilTag
# =========================
at_detector = Detector(families="tag36h11")
cap = cv2.VideoCapture(0)

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistorted = cv2.undistort(frame, MTX, DIST, None, NEW_MTX)

    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    results = at_detector.detect(gray)

    for det in results:
        u, v = det.center
        X, Y = get_real_world_xy(u, v)

        cx, cy = int(u), int(v)

        cv2.circle(undistorted, (cx, cy), 6, (0, 255, 0), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(undistorted, f"X:{X:.1f}", (cx + 20, cy - 20), font, 0.5, (0, 0, 255), 2)
        cv2.putText(undistorted, f"Y:{Y:.1f}", (cx + 20, cy + 20), font, 0.5, (0, 0, 255), 2)

    draw_world_points(undistorted)
    draw_boundary(undistorted)

    cv2.imshow("2D Plane Mapping", undistorted)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
