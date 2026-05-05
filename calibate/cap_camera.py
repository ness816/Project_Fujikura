import cv2
import os

# สร้างโฟลเดอร์ชื่อ images ถ้ายังไม่มี
if not os.path.exists('images'):
    os.makedirs('images')

# เปิดกล้อง (0 คือกล้องเว็บแคมหลัก)
cap = cv2.VideoCapture(0)

print("เริ่มการบันทึกภาพ...")
print("กด 's' เพื่อบันทึกภาพ (Save)")
print("กด 'q' เพื่อออกจากโปรแกรม (Quit)")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # แสดงหน้าจอ Preview
    cv2.imshow('Camera Calibration - Capture', frame)

    key = cv2.waitKey(1) & 0xFF
    
    # ถ้ากด 's' ให้บันทึกรูป
    if key == ord('s'):
        img_name = f"images/calib_img_{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"บันทึกรูปที่ {count} สำเร็จ: {img_name}")
        count += 1
    
    # ถ้ากด 'q' ให้ปิดโปรแกรม
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"เสร็จสิ้น! คุณได้ภาพทั้งหมด {count} ภาพ อยู่ในโฟลเดอร์ images")