import socket
import time

"""class MG400:
    def __init__(self, use_real_robot=False):
        self.use_real_robot = use_real_robot

        if self.use_real_robot:
            import socket

            self.s = socket.socket()
            self.s.connect(("192.168.1.6", 29999))
            print("[INFO] Connected to real robot")
        else:
            print("[INFO] Running in TEST MODE (no robot)")

    def moveL(self, x, y, z, r=0):
        if self.use_real_robot:
            cmd = f"MovL({x},{y},{z},{r})\n"
            self.s.send(cmd.encode())
        else:
            print(f"[MOCK MOVE] X={x:.2f}, Y={y:.2f}, Z={z:.2f}, R={r:.2f}")

    def close(self):
        if self.use_real_robot:
            self.s.close()"""


class MG400:
    def __init__(self, ip="192.168.1.6", use_real_robot=True):
        self.ip = ip
        self.use_real_robot = use_real_robot
        self.port = 29999  # พอร์ต Dashboard ที่ทดสอบสำเร็จ[cite: 1]

        if self.use_real_robot:
            try:
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client.connect((self.ip, self.port))
                self._prepare_robot()
                print(f"[INFO] Connected to MG400 at {self.ip}")
            except Exception as e:
                print(f"[ERROR] Connection failed: {e}")
                self.use_real_robot = False
                
    def set_speed(self, percentage):
        """ตั้งค่าความเร็วหุ่นยนต์ (1-100%)"""
        # ป้องกันการใส่ค่าเกินช่วงที่กำหนด
        percentage = max(1, min(percentage, 100))
        cmd = f"SpeedFactor({percentage})"
        print(f"[INFO] Setting robot speed to {percentage}%")
        return self.send_command(self.client, cmd) # ส่งคำสั่งผ่านพอร์ต 29999 พร้อม \n[cite: 1]

    def _prepare_robot(self):
        self.send_command(self.client, "ClearError()")
        time.sleep(0.5)
        self.send_command(self.client, "EnableRobot()")
        time.sleep(2)

    def send_command(self, sock, cmd):
        if not cmd.endswith("\n"):
            cmd += "\n"  # ใช้ Newline ตามโปรโตคอล Dashboard[cite: 1]
        sock.send(cmd.encode("utf-8"))
        time.sleep(0.1)
        return sock.recv(1024).decode("utf-8")

    def moveL(self, x, y, z, r=0):
        if self.use_real_robot:
            cmd = f"MovL({x},{y},{z},{r})"
            self.send_command(self.client, cmd)
        else:
            print(f"[MOCK MOVE] X={x:.2f}, Y={y:.2f}, Z={z:.2f}, R={r:.2f}")

    def set_gripper(self, state):
        cmd = f"DigitalOutputs(1,{state})"
        self.send_command(self.client, cmd)

    def close(self):
        if self.use_real_robot:
            self.send_command(self.client, "DisableRobot()")
            self.client.close()
