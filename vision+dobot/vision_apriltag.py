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

# =========================
# Detector
# =========================
at_detector = Detector(families="tag36h11")


# =========================
# pixel → world
# =========================
def get_real_world_xy(u, v):
    pixel_point = np.array([u, v, 1], dtype="float32").reshape(3, 1)
    real_point = np.dot(H, pixel_point)
    real_point /= real_point[2]
    return float(real_point[0]), float(real_point[1])


def draw_world_points(img):
    h_img, w_img = img.shape[:2]

    H_inv = np.linalg.inv(H)

    for x in range(-90, 91, 10):
        for y in range(-100, 101, 20):
            pt = np.array([x, y, 1], dtype="float32").reshape(3, 1)
            pix = np.dot(H_inv, pt)
            pix /= pix[2]

            px, py = int(pix[0]), int(pix[1])

            if 0 <= px < w_img and 0 <= py < h_img:
                if x == 0 and y == 0:
                    cv2.circle(img, (px, py), 4, (0, 0, 255), -1)
                    cv2.putText(
                        img,
                        "(0,0)",
                        (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.circle(img, (px, py), 2, (211, 0, 190), -1)


def draw_boundary(img):
    H_inv = np.linalg.inv(H)

    world_points = [(90, 100), (90, -100), (-90, -100), (-90, 100)]
    pixel_points = []

    for x, y in world_points:
        pt = np.array([x, y, 1], dtype="float32").reshape(3, 1)
        pix = np.dot(H_inv, pt)
        pix /= pix[2]

        pixel_points.append((int(pix[0]), int(pix[1])))

    for i in range(len(pixel_points)):
        cv2.line(img, pixel_points[i], pixel_points[(i + 1) % len(pixel_points)], (255, 0, 0), 2)


# =========================
# ใช้กับ main.py
# =========================
# ใน vision_apriltag.py (แก้ไขฟังก์ชันเดิม)
def get_all_targets(undistorted_frame):
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    results = at_detector.detect(gray)

    tag_list = []
    for det in results:
        u, v = det.center
        world_x, world_y = get_real_world_xy(u, v)
        tag_list.append({"world": (world_x, world_y), "pixel": (int(u), int(v))})
    return tag_list


# =========================
# DEBUG MODE
# =========================
def debug_view():
    cap = cv2.VideoCapture(0)

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

            cv2.putText(
                undistorted,
                f"X:{X:.1f}",
                (cx + 20, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                undistorted,
                f"Y:{Y:.1f}",
                (cx + 20, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        draw_world_points(undistorted)
        draw_boundary(undistorted)

        cv2.imshow("Vision Debug", undistorted)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# RUN DEBUG
# =========================
if __name__ == "__main__":
    debug_view()
