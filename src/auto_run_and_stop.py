import time
import sys
import threading

# ฟังก์ชันอ่าน input พร้อม timeout
def get_user_input(timeout):
    user_input = [None]

    def read_input():
        user_input[0] = input("\n⏳ Type 'stop' within 10 seconds to stop: \n")

    input_thread = threading.Thread(target=read_input)
    input_thread.start()
    input_thread.join(timeout)  # รอ input ตามเวลาที่กำหนด

    return user_input[0]  # คืนค่าข้อความที่พิมพ์ หรือ None ถ้า timeout

# เริ่มต้นตัวแปร
count = 0
max_iterations = 10000

# วนลูปบวกเลข
while count < max_iterations:
    count += 1  # บวกเลข +1
    print(f"🔢 Count: {count}")

    # ทุกๆ 3 ครั้ง ให้หยุดรอ input 10 วินาที
    if count % 3 == 0:
        user_input = get_user_input(5)

        if user_input and user_input.lower() == "stop":
            print("\n🛑 Program Stopped by User")
            sys.exit()  # ออกจากโปรแกรม

print("\n✅ Completed 10,000 iterations!")
