import winsound
import time

good_val = 1

while good_val !=  0:
    winsound.Beep(1000,400)

    if 0xff == ord('q'):
        break