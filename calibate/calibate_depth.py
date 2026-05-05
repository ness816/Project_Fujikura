import os

import cv2
import numpy as np
from pupil_apriltags import Detector


def main():
    # --- 1. การตั้งค่าพารามิเตอร์ ---
    TAG_SIZE = 0.02  # แก้ไขขนาด Tag จริงตรงนี้ (หน่วยเมตร)
    NPZ_FILE = "camera_data.npz"

    # --- 2. โหลดไฟล์ Calibration (.npz) ---
    if not os.path.exists(NPZ_FILE):
        print(f"❌ ไม่พบไฟล์ {NPZ_FILE}")
        return

    try:
        with np.load(NPZ_FILE) as data:
            mtx = data["mtx"]
            dist = data["dist"]

            # ดึงพารามิเตอร์พื้นฐาน
            fx = mtx[0, 0]
            fy = mtx[1, 1]
            cx = mtx[0, 2]
            cy = mtx[1, 2]
            print(f"✅ โหลดค่า Calibration สำเร็จ")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # --- 3. เริ่มต้นระบบ ---
    at_detector = Detector(families="tag36h11")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องได้")
        return

    # เตรียมพารามิเตอร์สำหรับ Undistort ครั้งเดียวเพื่อประหยัด CPU
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        # อัปเดตพารามิเตอร์กล้องตาม New Camera Matrix เพื่อให้คำนวณระยะทางแม่นยำหลังแก้ภาพโค้ง
        fx_new = new_camera_mtx[0, 0]
        fy_new = new_camera_mtx[1, 1]
        cx_new = new_camera_mtx[0, 2]
        cy_new = new_camera_mtx[1, 2]
        camera_params_new = [fx_new, fy_new, cx_new, cy_new]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- แก้ความโค้งของภาพ (Undistort) ---
        undistorted_img = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)

        # ตัดภาพตาม ROI (ถ้าต้องการตัดขอบดำที่เกิดจากการ Undistort)
        # x, y, w_roi, h_roi = roi
        # undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi]

        # แปลงเป็นขาวดำจากภาพที่แก้ความโค้งแล้ว
        gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)

        # ตรวจจับ AprilTag โดยใช้พารามิเตอร์ใหม่ที่แมพกับภาพ Undistorted
        results = at_detector.detect(
            gray, estimate_tag_pose=True, camera_params=camera_params_new, tag_size=TAG_SIZE
        )

        for r in results:
            (cx_px, cy_px) = (int(r.center[0]), int(r.center[1]))
            cz_mm = r.pose_t[2][0] * 1000

            # วาดเส้นขอบ
            for i in range(4):
                pt1 = tuple(r.corners[i].astype(int))
                pt2 = tuple(r.corners[(i + 1) % 4].astype(int))
                cv2.line(undistorted_img, pt1, pt2, (0, 255, 0), 2)

            # แสดงข้อมูล
            distance_mm = (
                np.sqrt(r.pose_t[0][0] ** 2 + r.pose_t[1][0] ** 2 + r.pose_t[2][0] ** 2) * 1000
            )

            label = f"ID: {r.tag_id} | Euclid: {distance_mm:.1f} mm"
            cv2.putText(
                undistorted_img,
                label,
                (cx_px - 50, cy_px - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.circle(undistorted_img, (cx_px, cy_px), 5, (0, 0, 255), -1)

        cv2.imshow("Undistorted AprilTag Depth", undistorted_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
