import cv2
import numpy as np
from robot_control import MG400
from vision_apriltag import DIST, MTX, NEW_MTX, draw_boundary, draw_world_points, get_all_targets

# --- ตัวแปรสำหรับสถานะระบบ ---
selected_index = -1  # เก็บลำดับของ Tag ที่ถูกเลือก (-1 คือยังไม่เลือก)
current_tags = []  # เก็บข้อมูล Tag ทั้งหมดที่เจอในเฟรมปัจจุบัน


# --- ฟังก์ชันจัดการการคลิกเมาส์ (เงื่อนไขข้อ 3) ---
def select_tag_event(event, x, y, flags, param):
    global selected_index, current_tags
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = 30  # รัศมีการคลิก (พิกเซล)
        selected_index = -1  # รีเซ็ตการเลือกก่อน
        for i, tag in enumerate(current_tags):
            tx, ty = tag["pixel"]
            # คำนวณระยะห่างระหว่างจุดที่คลิกกับกลาง Tag
            dist = np.sqrt((x - tx) ** 2 + (y - ty) ** 2)
            if dist < min_dist:
                selected_index = i
                print(f"[UI] Selected Tag #{i}")
                break


# ตั้งค่าเริ่มต้น
cap = cv2.VideoCapture(0)
robot = MG400(use_real_robot=False)
cv2.namedWindow("Vision Main Control")
cv2.setMouseCallback("Vision Main Control", select_tag_event)

print("--- ระบบพร้อมทำงาน ---")
print("1. คลิกที่ AprilTag เพื่อเลือกเป้าหมาย")
print("2. กด SPACE เพื่อเริ่มหยิบ")
print("3. กด Q เพื่อออกจากโปรแกรม")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # แก้ไขภาพเบี้ยว
    undistorted = cv2.undistort(frame, MTX, DIST, None, NEW_MTX)
    draw_world_points(undistorted)
    draw_boundary(undistorted)

    # รับข้อมูล Tag ทั้งหมด
    current_tags = get_all_targets(undistorted)

    # วาด Tag ทั้งหมดที่เจอ

    for i, tag in enumerate(current_tags):
        tx, ty = tag["pixel"]  # พิกัดบนหน้าจอ
        wx, wy = tag["world"]  # พิกัดจริงที่คำนวณได้

        # กำหนดสี: ถ้าเลือกอยู่ให้เป็นสีเขียว ถ้าไม่ใช่ให้เป็นสีขาว
        color = (0, 255, 0) if i == selected_index else (255, 255, 255)
        thickness = 2 if i == selected_index else 1

        # วาดจุดกึ่งกลางและลำดับ Index
        cv2.circle(undistorted, (tx, ty), 8, color, -1)
        cv2.putText(
            undistorted,
            f"#{i}",
            (tx + 10, ty - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness,
        )

        # --- ส่วนที่เพิ่มใหม่: แสดงพิกัด X, Y ของทุกอัน ---
        coord_text = f"X:{wx:.1f} Y:{wy:.1f}"
        cv2.putText(
            undistorted, coord_text, (tx + 10, ty + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # --- เช็คเงื่อนไขและเตรียมหยิบ ---
    if selected_index != -1 and selected_index < len(current_tags):
        target = current_tags[selected_index]
        X, Y = target["world"]

        # 1. เงื่อนไขขอบเขต (Boundary Check)
        if -90 <= X <= 90 and -100 <= Y <= 100:
            status_text = f"Target Ready: X={X:.1f} Y={Y:.1f} | Press SPACE"
            status_color = (0, 255, 0)  # เขียว

            # 2. เงื่อนไขการกด Spacebar (Trigger)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                print(f"[ACTION] Picking Tag at X:{X:.2f}, Y:{Y:.2f}")
                robot.moveL(X, Y, 50, 0)
        else:
            status_text = "OUT OF BOUNDS (นอกกรอบการทำงาน)"
            status_color = (0, 0, 255)  # แดง

        cv2.putText(
            undistorted, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
        )
    else:
        cv2.putText(
            undistorted,
            "Please click on a Tag to select",
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
cv2.destroyAllWindows()
        