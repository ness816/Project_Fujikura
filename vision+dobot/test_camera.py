import cv2

# 0 คือ ID ของกล้องตัวหลัก (ถ้ามีหลายตัวอาจจะเป็น 1, 2, ...)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

while True:
    # อ่านเฟรมจากกล้อง
    ret, frame = cap.read()

    if not ret:
        print("ไม่สามารถรับเฟรมจากกล้องได้ (Stream end?)")
        break

    # แสดงผลในหน้าต่างที่ชื่อว่า 'Camera'
    cv2.imshow("Camera", frame)

    # รอการกดปุ่ม 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# คืนทรัพยากรให้ระบบ
cap.release()
cv2.destroyAllWindows()
