import winsound
import time

for i in range(3):
    print(f"Beep {i+1}")
    winsound.Beep(1000, 400)
    time.sleep(2)
