#!/usr/bin/env python3
import rospy
from bvi_fyp.srv import Navigate
from std_msgs.msg import Bool

class Main:
    def __init__(self):
        rospy.init_node("main")

        self.beep_pub = rospy.Publisher("/beep_control", Bool, queue_size=1, latch = True)
        rospy.loginfo("in main node")
        # self.beep_pub.publish(True)
        rospy.loginfo("publish true")


def go_to_checkpoint(checkpoint_name):
    rospy.wait_for_service('navigate')
    try:
        nav_service = rospy.ServiceProxy('navigate', Navigate)
        response = nav_service(checkpoint_name)
        print(f"Reached: {response.reach}, Message: {response.message}")
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

if __name__ == "__main__":
    Main()
    rospy.loginfo("run main ard")
    rospy.spin()

    # Change this to desired checkpoint
    checkpoint = "shelf"

    go_to_checkpoint(checkpoint)
