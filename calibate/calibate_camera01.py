#Camera Matrix ($mtx$): บอกว่ากล้องของคุณมีทางยาวโฟกัสเท่าไหร่ และจุดกึ่งกลางภาพอยู่ที่ไหน
#Distortion Coefficients ($dist$): บอกว่าเลนส์ของคุณเบี้ยวแบบ "ถังเบียร์" (Radial) หรือ "คางหมู" (Tangential) แค่ไหน
import glob
import cv2
import numpy as np

# ตั้งค่าจุดตัดภายใน (กระดาน 8x8 ช่อง จะมีจุดตัด 7x7)
CHECKERBOARD = (7, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# เตรียมพิกัด 3D
objpoints = []
imgpoints = []

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)

# ดึงรูปจากโฟลเดอร์ images ที่คุณบันทึกไว้
images = glob.glob("images/*.jpg")

print(f"กำลังประมวลผลภาพทั้งหมด {len(images)} ภาพ...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ค้นหาจุดตัด
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        objpoints.append(objp)
        # ปรับความแม่นยำจุดตัด
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # วาดจุดให้ดูว่าหาเจอไหม
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Checking Corners", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# คำนวณหาค่า Camera Matrix และ Distortion Coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

if ret:
    print("\n--- คาลิเบตเสร็จสมบูรณ์! ---")
    print("1. Camera Matrix (ค่าพารามิเตอร์ภายใน):")
    print(mtx)
    print("\n2. Distortion Coefficients (ค่าความบิดเบี้ยว):")
    print(dist)

    # บันทึกค่าลงไฟล์ .npz เพื่อเอาไปใช้ในโปรเจกต์อื่นโดยไม่ต้องคาลิเบตใหม่
    np.savez("camera_data.npz", mtx=mtx, dist=dist)
    print("\nบันทึกค่าลงไฟล์ 'camera_data.npz' เรียบร้อยแล้ว!")
else:
    print("ไม่สามารถคาลิเบตได้ กรุณาเช็คว่ารูปในโฟลเดอร์ images ชัดเจนพอหรือไม่")
