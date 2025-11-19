#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import ultralytics
from datetime import datetime
import os
from bvi_fyp.srv import stt, sttRequest
from bvi_fyp.srv import tts, ttsRequest

class CaneDetection:
    def __init__(self):
        rospy.init_node('cane_detection')

        # ========= FLAGS & STATE FIRST =========
        self.stop_on_cane = False
        self.CONFIDENCE_THRESHOLD = 0.7
        self.cv_image = None

        # BVI target state
        self.bvi_target_box = None   # (x1, y1, x2, y2)
        self.follow_started = False

        # Save directory
        self.save_dir = "/home/mustar/catkin_ws/src/bvi_fyp/src/image_detected"
        os.makedirs(self.save_dir, exist_ok=True)

        # ========= MODELS & BRIDGE =========
        self.model_cane = ultralytics.YOLO('/home/mustar/catkin_ws/src/bvi_fyp/src/model/cane_best.pt')
        self.model_person = ultralytics.YOLO('yolov8n.pt')  # COCO model
        self.bridge = CvBridge()

        # ========= SPEECH SERVICES (before subscriber!) =========
        rospy.loginfo("Waiting for /speech_to_text, /text_to_speech")
        rospy.wait_for_service("/speech_to_text")
        rospy.wait_for_service("/text_to_speech")

        self.sr_client = rospy.ServiceProxy("/speech_to_text", stt)
        self.tts_client = rospy.ServiceProxy("/text_to_speech", tts)

        rospy.loginfo("Speech services ready.")
        rospy.loginfo("Continuous cane detection node started.")

        # 🔊 Intro message
        self.speak(
            "Hello. I am your faculty navigation guide robot. "
            "I am now scanning for blind or visually impaired individuals with a guide cane. "
        )

        # ========= SUBSCRIBER CREATED LAST =========
        image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)

    def speak(self, text):
        try:
            self.tts_client(ttsRequest(text=text))
        except Exception as e:
            rospy.logerr(f"TTS call failed: {e}")

    def listen(self):
        try:
            resp = self.sr_client(sttRequest())
            if resp.success and resp.text.strip():
                return resp.text.strip()
            return None
        except Exception as e:
            rospy.logerr(f"SR call failed: {e}")
            return None

    def start_follow_target(self, person_box):
        """
        Placeholder: logic to start follow-me service for this specific BVI person.
        person_box: (x1, y1, x2, y2) in image coordinates.
        """
        x1, y1, x2, y2 = person_box
        cx = int(0.5 * (x1 + x2))
        cy = int(0.5 * (y1 + y2))

        rospy.loginfo(f"[FOLLOW-ME] BVI target selected at image center: ({cx}, {cy})")
        rospy.loginfo("[FOLLOW-ME] Here you would call your follow-me service with this target.")
        

        # Example (pseudo-code):
        # req = FollowMeRequest()
        # req.target_x = cx
        # req.target_y = cy
        # resp = self.follow_me_client(req)

        self.follow_started = True
        os.system("rosrun bvi_fyp voice_qa.py")

    def image_callback(self, msg_color):

        # If we already decided to stop detection (and follow-me is running), just return
        if self.stop_on_cane:
            return

        # Convert image
        try:
            img_color = self.bridge.imgmsg_to_cv2(msg_color, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn(str(e))
            return

        frame = np.flip(img_color, axis=1)
        self.cv_image = frame.copy()

        # Run both models
        results_cane = self.model_cane(self.cv_image)
        cane_boxes = results_cane[0].boxes

        results_person = self.model_person(self.cv_image)
        person_boxes = results_person[0].boxes

        persons = []
        canes = []

        # -------------------------
        # PROCESS PERSON DETECTIONS
        # -------------------------
        for box in person_boxes:
            conf = box.conf.item()
            cls_id = int(box.cls.item())

            if conf >= self.CONFIDENCE_THRESHOLD and cls_id == 0:  # class 0 = person
                xyxy = box.xyxy.numpy().flatten()
                x1, y1, x2, y2 = map(float, xyxy)
                persons.append((x1, y1, x2, y2))

                # Draw green box
                cv2.rectangle(self.cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(self.cv_image, f"person {conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # -------------------------
        # PROCESS CANE DETECTIONS
        # -------------------------
        for box in cane_boxes:
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            class_name = results_cane[0].names[cls_id].lower()

            if conf >= self.CONFIDENCE_THRESHOLD and class_name == "guide cane":
                xyxy = box.xyxy.numpy().flatten()
                x1, y1, x2, y2 = map(float, xyxy)
                canes.append((x1, y1, x2, y2))

                # Draw blue box
                cv2.rectangle(self.cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                cv2.putText(self.cv_image, f"cane {conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # -------------------------
        # CHECK PERSON WITH CANE
        # -------------------------
        person_with_cane = False
        bvi_person_box = None  # NEW: store the exact person we want to follow

        if persons and canes:
            for (px1, py1, px2, py2) in persons:
                for (cx1, cy1, cx2, cy2) in canes:
                    cane_cx = 0.5 * (cx1 + cx2)
                    cane_cy = 0.5 * (cy1 + cy2)

                    # cane must be horizontally inside person body
                    if (px1 <= cane_cx <= px2) and (cane_cy >= py1):
                        person_with_cane = True
                        bvi_person_box = (px1, py1, px2, py2)
                        break
                if person_with_cane:
                    break

        # -------------------------
        # SAVE, SELECT TARGET & START FOLLOW
        # -------------------------
        if person_with_cane and not self.follow_started:
            rospy.loginfo("PERSON WITH CANE DETECTED! Saving and starting follow-me logic...")

            self.speak(
                    "Excuse me. I have detected a person with a guide cane. "
                    "Can you please stop walking and stand still? I will approach you now."
                )

            # Save image (same as before)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{self.save_dir}/person_cane_{timestamp}.jpg"
            cv2.imwrite(save_path, self.cv_image)
            rospy.loginfo(f"Saved image at {save_path}")

            # NEW: store this person as the BVI target and start follow-me
            self.bvi_target_box = bvi_person_box
            self.start_follow_target(self.bvi_target_box)
            print("target bvi individual coordinate: ", px1, px2, py1, py2)

            # If you want to completely stop this node’s detection after that:
            self.stop_on_cane = True
            return

        # Show detection
        cv2.imshow("Cane Detection", self.cv_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    try:
        CaneDetection()
        rospy.spin()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down continuous detection node.")
