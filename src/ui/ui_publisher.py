import rospy
import json
import time
from std_msgs.msg import String

class UIPublisher:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UIPublisher, cls).__new__(cls)
        return cls._instance

    def init(self):
        # Call this AFTER your node's rospy.init_node()
        if getattr(self, "_inited", False):
            return

        self.state_pub = rospy.Publisher("/ui_state", String, queue_size=10, latch=True)
        self.chat_pub  = rospy.Publisher("/ui_chat_log", String, queue_size=50)
        self._inited = True
        rospy.loginfo("UIPublisher ready: /ui_state, /ui_chat_log")

    def publish_state(self, **kwargs):
        msg = String()
        msg.data = json.dumps({"ts": time.time(), **kwargs})
        self.state_pub.publish(msg)

    def publish_chat(self, role, text):
        msg = String()
        msg.data = json.dumps({"ts": time.time(), "role": role, "text": text})
        self.chat_pub.publish(msg)
