#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
from sound_play.libsoundplay import SoundClient
import time

class BeepNode:
    def __init__(self):
        rospy.init_node("beep_node")
        rospy.loginfo("Starting beep node (auditory cue)")

        # Sound client
        self.sound_client = SoundClient()
        rospy.sleep(1.0)  # allow sound_play to initialize

        # Beep control
        self.beep_active = False
        self.beep_interval = rospy.get_param("~beep_interval", 0.6)  # seconds
        self.last_beep_time = 0.0

        # Subscriber
        self.sub = rospy.Subscriber("/beep_control", Bool, self.beep_callback)

        # Timer to run beep periodically
        self.timer = rospy.Timer(rospy.Duration(0.05), self.timer_callback)  # 20 Hz

        rospy.loginfo("Beep node ready. Waiting for commands.")
        rospy.spin()  # keep node alive

    def beep_callback(self, msg):
        self.beep_active = msg.data  # must use msg.data, not msg
        state = "ON" if self.beep_active else "OFF"
        rospy.loginfo(f"Beep turned {state}")

    def timer_callback(self, event):
        if self.beep_active:
            now = time.time()
            if now - self.last_beep_time >= self.beep_interval:
                self.sound_client.playWave('/home/mustar/catkin_ws/src/bvi_fyp/src/navigation/beep.wav')
                self.last_beep_time = now

if __name__ == "__main__":
    try:
        BeepNode()
    except rospy.ROSInterruptException:
        pass
