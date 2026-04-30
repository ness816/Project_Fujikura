import socket
import time


class MG400:
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
            self.s.close()
