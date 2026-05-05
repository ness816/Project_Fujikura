import cv2

import numpy as np

from pupil_apriltags import Detector



# =========================

# 1. โหลด Intrinsic

# =========================

try:

    calib_data = np.load("camera_data.npz")

    mtx = calib_data["mtx"]

    dist = calib_data["dist"]

    print("[ระบบ] โหลดค่า Intrinsic สำเร็จ")

except Exception as e:

    print(f"[ข้อผิดพลาด] {e}")

    exit()



# =========================

# 2. กำหนดพิกัดโลกจริง (mm)

# (0,0 = กลางพื้นที่)

# =========================

REAL_COORDS = np.array(

    [

        [-90, 100],  # ID 1 (บนซ้าย)

        [90, 100],  # ID 2 (บนขวา)

        [90, -100],  # ID 3 (ล่างขวา)

        [-90, -100],  # ID 4 (ล่างซ้าย)

    ],

    dtype="float32",

)



TARGET_IDS = [1, 2, 3, 4]



at_detector = Detector(families="tag36h11")

cap = cv2.VideoCapture(0)



print("=== Calibration (ใช้ corner ด้านในอัตโนมัติ) ===")

print("กด 'S' เพื่อบันทึก | กด 'Q' เพื่อออก")



# =========================

# LOOP

# =========================

while True:

    ret, frame = cap.read()

    if not ret:

        break



    # --- Undistort ---

    h, w = frame.shape[:2]

    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)



    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    results = at_detector.detect(gray)



    # จุดกลางภาพ

    cx, cy = w // 2, h // 2



    detected_tags = {}



    # =========================

    # เลือก corner ที่ใกล้ center (ด้านใน)

    # =========================

    for res in results:

        tid = res.tag_id



        if tid not in TARGET_IDS:

            continue



        corners = res.corners



        max_dist = -1

        best_corner = None



        for c in corners:

            dx = c[0] - cx

            dy = c[1] - cy

            dist2 = dx * dx + dy * dy



            if dist2 > max_dist:

                max_dist = dist2

                best_corner = c



        detected_tags[tid] = best_corner



        # วาดจุด

        pt = tuple(best_corner.astype(int))

        cv2.circle(undistorted, pt, 8, (0, 255, 0), -1)

        cv2.putText(

            undistorted,

            f"ID:{tid}",

            pt,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 0, 255),

            2,

        )



    cv2.imshow("Calibration", undistorted)

    key = cv2.waitKey(1) & 0xFF



    # =========================

    # กด S เพื่อคาลิเบรต

    # =========================

    if key == ord("s"):

        if all(tid in detected_tags for tid in TARGET_IDS):

            pixel_points = np.array([detected_tags[tid] for tid in TARGET_IDS], dtype="float32")



            H, _ = cv2.findHomography(pixel_points, REAL_COORDS)



            np.savez(

                "robot_vision_config_clean_new.npz",

                mtx=mtx,

                dist=dist,

                new_mtx=new_mtx,

                homography=H,

            )



            print("\n[สำเร็จ] คาลิเบรตเสร็จ! → robot_vision_config_clean_new.npz")

            break



        else:

            missing = [tid for tid in TARGET_IDS if tid not in detected_tags]

            print(f"[แจ้งเตือน] ยังขาด ID: {missing}")



    elif key == ord("q"):

        break



cap.release()

cv2.destroyAllWindows()