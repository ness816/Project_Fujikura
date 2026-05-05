import cv2
import numpy as np
from pupil_apriltags import Detector

# 1. โหลดค่า Intrinsic (แก้ชื่อไฟล์ให้ตรงกับที่คุณมี)
try:
    calib_data = np.load("camera_data.npz")
    mtx = calib_data["mtx"]
    dist = calib_data["dist"]
    print("[ระบบ] โหลดค่า Intrinsic สำเร็จ")
except Exception as e:
    print(f"[ข้อผิดพลาด] ไม่สามารถโหลดไฟล์ camera_data.npz ได้: {e}")
    exit()

# 2. ตั้งค่าพิกัดโลกจริง (กึ่งกลางพื้นที่ 180x200 มม. คือ 0,0)
# ปรับลำดับให้ตรงกับรูปถ่ายของคุณ: ID 1(บนซ้าย), 2(บนขวา), 3(ล่างขวา), 4(ล่างซ้าย)
REAL_COORDS = np.array(
    [
        [-90, 100],  # พิกัดจริงสำหรับ ID 1 (บนซ้าย) 90 100
        [90, 100],  # พิกัดจริงสำหรับ ID 2 (บนขวา)
        [90, -100],  # พิกัดจริงสำหรับ ID 3 (ล่างขวา)
        [-90, -100],  # พิกัดจริงสำหรับ ID 4 (ล่างซ้าย)
    ],
    dtype="float32",
)

at_detector = Detector(families="tag36h11")
cap = cv2.VideoCapture(0)  # เปลี่ยนเป็น 1 หรือ 2 ตามพอร์ตกล้อง Jetson

print("--- โปรแกรมคาลิเบรตตำแหน่ง (ID 1, 2, 3, 4) ---")
print("กด 'S' เพื่อบันทึกค่าเมื่อตัวเลขขึ้นครบทุกมุม | กด 'Q' เพื่อออก")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- แก้ความโค้งของภาพ (Undistort) ---
    h, w = frame.shape[:2]
    # ใช้ alpha=0 เพื่อตัดขอบดำทิ้งและซูมภาพให้พอดีหน้าจอ
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    undistorted_img = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)

    gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    results = at_detector.detect(gray)

    # เก็บพิกัดจุดกึ่งกลางของ Tag ที่ตรวจพบ
    detected_tags = {res.tag_id: res.center for res in results}

    # วาดข้อมูลบนหน้าจอ
    for tid, center in detected_tags.items():
        if tid in [1, 2, 3, 4]:
            cv2.circle(undistorted_img, tuple(center.astype(int)), 8, (0, 255, 0), -1)
            cv2.putText(
                undistorted_img,
                f"ID:{tid}",
                tuple(center.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    cv2.imshow("Calibration - Undistorted View", undistorted_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        # ตรวจสอบว่าเจอ ID 1, 2, 3, 4 ครบไหม
        target_ids = [1, 2, 3, 4]
        if all(tid in detected_tags for tid in target_ids):
            # เรียงพิกัดพิกเซลตามลำดับพิกัดจริงที่ตั้งไว้
            pixel_points = np.array([detected_tags[tid] for tid in target_ids], dtype="float32")

            # คำนวณหา Homography Matrix (H)
            H, _ = cv2.findHomography(pixel_points, REAL_COORDS)

            # บันทึกค่าทั้งหมดลงในไฟล์เดียวเพื่อเอาไปใช้ในโปรเจกต์อื่น
            np.savez(
                "robot_vision_config_cornor.npz",
                mtx=mtx,
                dist=dist,
                new_mtx=new_camera_mtx,
                homography=H,
            )

            print("\n[สำเร็จ] คาลิเบรตเสร็จสิ้น! บันทึกไฟล์ 'robot_vision_config_new_cornor.npz'")
            break
        else:
            missing = [tid for tid in target_ids if tid not in detected_tags]
            print(f"[แจ้งเตือน] ยังขาด ID: {missing}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
