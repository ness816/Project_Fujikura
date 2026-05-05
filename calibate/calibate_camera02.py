# ภาพที่เบี้ยวจากเลนส์ กับ ภาพที่ถูกแก้ไขแล้ว
import cv2
import numpy as np

# โหลดค่าที่คำนวณไว้
with np.load("camera_data.npz") as data:
    mtx = data["mtx"]
    dist = data["dist"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # แก้ไขภาพเบี้ยว
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # 3. ตัดขอบ (Crop) ตามค่า ROI ที่คำนวณได้
    x, y, w_roi, h_roi = roi
    dst = dst[y : y + h_roi, x : x + w_roi]

    # แสดงผลเทียบกัน
    cv2.imshow("Original (Before)", frame)
    #cv2.imshow("Calibrated (After)", dst)

    # 4. แสดงผลภาพที่ตัดแล้ว
    cv2.imshow("Final Result (Cropped)", dst)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
