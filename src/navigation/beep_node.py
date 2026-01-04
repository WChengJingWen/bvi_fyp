
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
            queue_size=1
        )

        rospy.loginfo("Beep node ready. Waiting for commands.")

    def beep_callback(self, msg):
        self.beep_active = msg.data
        state = "ON" if self.beep_active else "OFF"
        rospy.loginfo(f"Beep turned {state}")

    def run(self):
        rate = rospy.Rate(20)  # loop rate

        while not rospy.is_shutdown():
            if self.beep_active:
                now = time.time()
                if now - self.last_beep_time >= self.beep_interval:
                    self.sound_client.beep()
                    self.last_beep_time = now

            rate.sleep()


if __name__ == "__main__":
    try:
        node = BeepNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
