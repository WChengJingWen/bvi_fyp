#!/usr/bin/env python3
import rospy
import json
import time
from std_msgs.msg import String

STATE_FILE = "/home/mustar/catkin_ws/src/bvi_fyp/src/tmp/ui_state.json"
CHAT_FILE  = "/home/mustar/catkin_ws/src/bvi_fyp/src/tmp/ui_chat.jsonl"   
DIST_FILE = "/home/mustar/catkin_ws/src/bvi_fyp/src/tmp/ui_distance.jsonl"

def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    # atomic replace
    import os
    os.replace(tmp, path)

def on_state(msg):
    try:
        data = json.loads(msg.data)
    except Exception:
        # if already dict-like string, ignore parse error
        return

    data["_received_ts"] = time.time()
    write_json(STATE_FILE, data)

def on_chat(msg):
    try:
        data = json.loads(msg.data)
    except Exception:
        return

    data["_received_ts"] = time.time()
    with open(CHAT_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")

def on_distance(msg):
    try:
        data = json.loads(msg.data)
    except Exception:
        return

    data["_received_ts"] = time.time()
    with open(DIST_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")

def main():
    rospy.init_node("ui_bridge")

    # 🔥 Clear chat history on each run
    open(CHAT_FILE, "w").close()

    rospy.Subscriber("/ui_state", String, on_state, queue_size=10)
    rospy.Subscriber("/ui_chat_log", String, on_chat, queue_size=50)
    rospy.Subscriber("/ui_distance", String, on_distance, queue_size=50)
    rospy.loginfo("ui_bridge running: writing %s and %s", STATE_FILE, CHAT_FILE)
    rospy.spin()


if __name__ == "__main__":
    main()
