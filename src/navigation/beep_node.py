
# sudo apt install ros-noetic-sound-play

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
        self.sub = rospy.Subscriber(
            "/beep_control",
            Bool,
            self.beep_callback,
        )

        rospy.loginfo("Beep node ready. Waiting for commands.")
        self.run()

    def beep_callback(self, msg):
        self.beep_active = msg
        state = "ON" if self.beep_active else "OFF"
        rospy.loginfo(f"Beep turned {state}")

    def run(self):
        rospy.loginfo("in run()")
        rate = rospy.Rate(20)  # loop rate

        while not rospy.is_shutdown():
            # rospy.loginfo("in beep run while loop")
            if self.beep_active:
                # rospy.loginfo("beep active ard")
                now = time.time()
                if now - self.last_beep_time >= self.beep_interval:
                    self.sound_client.playWave('/home/mustar/catkin_ws/src/bvi_fyp/src/navigation/beep.wav')
                    self.last_beep_time = now

            rate.sleep()

if __name__ == "__main__":
    try:
        BeepNode()
        rospy.loginfo("beep node run ard")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
