import winsound
import time

class Posture_Alert:
    def __init__(self, frequency = 1000, duration = 400):
        self.frequency = frequency
        self.duration = duration

        self.alert_fired = False
        self.bad_posture_start = None
        self.good_posture_start = None

    def update(self, good_posture):
        now = time.time()

        if good_posture:
            self.bad_posture_start = None
            self.alert_fired = False
            
        else:
            if self.bad_posture_start is None:
                self.bad_posture_start = now
            
            if not self.alert_fired:
                winsound.Beep(self.frequency,self.duration)    
                self.alert_fired = True

    
    def bad_duration(self):
        if self.bad_posture_start is None:
            return 0
        return  int(time.time() - self.bad_posture_start)
    