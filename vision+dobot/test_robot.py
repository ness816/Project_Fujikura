import socket
import time

HOST = "192.168.1.6"  # IP ของ MG400
PORT = 29999  # Control port

# สร้าง socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))


def send(cmd):
    client.send((cmd + "\n").encode())
    time.sleep(0.1)


# เปิดใช้งานหุ่น
send("EnableRobot()")

time.sleep(1)

# เคลื่อนที่แบบ Joint (MovJ)
# format: MovJ(x, y, z, r)
send("MovJ(350, 0, 0, 0)")

print("Sent command!")

client.close()
