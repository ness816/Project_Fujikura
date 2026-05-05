import time

import cv2
import numpy as np
from robot_control import MG400
from vision_apriltag import DIST, MTX, NEW_MTX, draw_boundary, draw_world_points, get_all_targets

# --- ตั้งค่าคงที่สำหรับ Robot ---
SAFE_Z = -60  # ความสูงขณะเคลื่อนที่ผ่าน (มม.)
PICK_Z = -127.0  # ความสูงขณะหยิบวัตถุ (ต้องปรับตามความสูงจริงของพื้นโต๊ะ)
ROBOT_IP = "192.168.1.6"

# --- ตัวแปรสำหรับสถานะระบบ ---
selected_index = -1
current_tags = []

# เพิ่มส่วนนี้
OFFSET_X = 324.75  # 325.894
OFFSET_Y = -21.75  # 15.2473

PLACE_X = 280.1883
PLACE_Y = 212.0144

WAIT_X = 221.2044
WAIT_Y = -8.2583


def select_tag_event(event, x, y, flags, param):
    global selected_index, current_tags
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = 30
        selected_index = -1
        for i, tag in enumerate(current_tags):
            tx, ty = tag["pixel"]
            dist = np.sqrt((x - tx) ** 2 + (y - ty) ** 2)
            if dist < min_dist:
                selected_index = i
                print(f"[UI] Selected Tag #{i}")
                break


# --- เริ่มต้นระบบ ---
cap = cv2.VideoCapture(0)
# เปลี่ยน use_real_robot=True เมื่อต้องการเชื่อมต่อหุ่นจริง
robot = MG400(ip=ROBOT_IP, use_real_robot=True)

# ตั้งค่าความเร็วเป็น 50% เพื่อความปลอดภัยในการทดสอบครั้งแรก[cite: 1]
robot.set_speed(25)

cv2.namedWindow("Vision Main Control")
cv2.setMouseCallback("Vision Main Control", select_tag_event)

print("--- ระบบพร้อมทำงาน (Real Robot Mode) ---")
print(f"เชื่อมต่อหุ่นยนต์ที่ IP: {ROBOT_IP}")
print("1. คลิกที่ AprilTag เพื่อเลือกเป้าหมาย")
print("2. กด SPACE เพื่อเริ่มกระบวนการหยิบ")
print("3. กด Q เพื่อออกจากโปรแกรม")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Image Pre-processing
    undistorted = cv2.undistort(frame, MTX, DIST, None, NEW_MTX)
    draw_world_points(undistorted)
    draw_boundary(undistorted)

    # 2. Vision Detection[cite: 1]
    current_tags = get_all_targets(undistorted)

    # 3. วาด UI และสถานะ Tag
    for i, tag in enumerate(current_tags):
        tx, ty = tag["pixel"]
        wx, wy = tag["world"]
        color = (0, 255, 0) if i == selected_index else (255, 255, 255)

        cv2.circle(undistorted, (tx, ty), 8, color, -1)
        cv2.putText(
            undistorted,
            f"#{i} (X:{wx:.1f}, Y:{wy:.1f})",
            (tx + 10, ty - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    # 4. ตรวจสอบเงื่อนไขและสั่งการ
    if selected_index != -1 and selected_index < len(current_tags):
        target = current_tags[selected_index]
        raw_X, raw_Y = target["world"]  # ค่าดิบจาก Vision[cite: 3]

        # --- ส่วนที่ต้องเพิ่ม: คำนวณพิกัดหุ่นยนต์จริง ---
        X = raw_X + OFFSET_X
        Y = raw_Y + OFFSET_Y

        # --- ส่วนที่ต้องแก้ไข: ปรับ Boundary ให้ตรงกับพิกัดหุ่นยนต์ ---
        # สมมติ Vision ทำงานในช่วง -90 ถึง 90 ดังนั้น Robot จะเป็น 260 ถึง 440 (350-90 ถึง 350+90)
        if 250 <= X <= 402 and -102 <= Y <= 101 and -130 <= SAFE_Z <= -60 and -130 <= PICK_Z <= -60:
            status_text = f"READY: X={X:.1f} Y={Y:.1f} | Press SPACE"
            status_color = (0, 255, 0)

            # เมื่อกด SPACE ตัวแปร X, Y ที่ส่งไปหา robot.moveL จะเป็นค่าที่บวก Offset แล้ว
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                # ทุกคำสั่ง robot.moveL(X, Y, ...) ด้านล่างนี้จะใช้ค่าที่บวก Offset แล้วโดยอัตโนมัติ
                # print(f"[ACTION] Moving to X:{X:.2f}, Y:{Y:.2f}")
                # robot.moveL(X, Y, SAFE_Z, 0)

                # Step 0: initialPose
                print("START")
                robot.moveL(WAIT_X, WAIT_Y, SAFE_Z, 0)
                time.sleep(1.0)

                # Step 1: ไปที่ตำแหน่งเหนือวัตถุ
                print(f"Moving to Approach position at: X={X:.2f}, Y={Y:.2f}, Z={SAFE_Z}")
                robot.moveL(X, Y, SAFE_Z, 0)
                time.sleep(1.5)  # รอหุ่นเคลื่อนที่ถึงจุด

                # Step 2: ลงไปหยิบ
                print(f"Moving to Approach position at: X={X:.2f}, Y={Y:.2f}, Z={PICK_Z}")
                robot.moveL(X, Y, PICK_Z, 0)
                time.sleep(1.0)

                # Step 3: Gripper หยิบ
                print("gripper close = Pick")
                robot.set_gripper(1)
                time.sleep(1.0)

                # Step 4: ยกขึ้นกลับไปที่ความสูงปลอดภัย
                print(f"Moving to Approach position at: X={X:.2f}, Y={Y:.2f}, Z={SAFE_Z}")
                robot.moveL(X, Y, SAFE_Z, 0)
                time.sleep(1.0)

                # Step 5: บนตำแหน่งวางที่เซ็ตไว้
                print(f"Moving to Approach position at: X={PLACE_X}, Y={PLACE_Y}, Z={SAFE_Z}")
                robot.moveL(PLACE_X, PLACE_Y, SAFE_Z, 0)
                time.sleep(1.0)

                # Step 6: วางตามตำแหน่งที่เซ็ตไว้
                print(f"Moving to Approach position at: X={PLACE_X}, Y={PLACE_Y}, Z={PICK_Z}")
                robot.moveL(PLACE_X, PLACE_Y, PICK_Z, 0)
                time.sleep(1.0)

                # Step 7: Gripper ปล่อย
                print("gripper open = Place")
                robot.set_gripper(0)
                time.sleep(1.0)

                # Step 8: บนตำแหน่งวางที่เซ็ตไว้
                print(f"Moving to Approach position at: X={PLACE_X}, Y={PLACE_Y}, Z={SAFE_Z}")
                robot.moveL(PLACE_X, PLACE_Y, SAFE_Z, 0)
                time.sleep(1.0)

                print("[ACTION] Sequence Completed.")

                # Step 0: initialPose
                print("WAITING FOR ORDERS")
                robot.moveL(WAIT_X, WAIT_Y, SAFE_Z, 0)
                time.sleep(1.0)

        else:
            status_text = "OUT OF BOUNDS !!!!!!!!"
            status_color = (0, 0, 255)

        cv2.putText(
            undistorted, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
        )
    else:
        cv2.putText(
            undistorted,
            "Please select a Tag",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
        )

    cv2.imshow("Vision Main Control", undistorted)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
robot.close()  # ปิดการเชื่อมต่อหุ่นยนต์
cv2.destroyAllWindows()


"""
ค่าที่ต้องไปเอาจากdobot studio 
PICK_Z
SAFE_Z
จดค่า X และ Y ต่ำสุด/สูงสุด
ทิศทางของแกน X และ Y ตรงกับทิศทาง Vision
IP Address ของหุ่นยนต์"""
