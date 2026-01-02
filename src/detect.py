#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import ultralytics
from datetime import datetime
import os
import threading
import time
from bvi_fyp.srv import stt, sttRequest
from bvi_fyp.srv import tts, ttsRequest
from ui.ui_publisher import UIPublisher

class CaneDetection:
    def __init__(self):
        rospy.init_node('cane_detection')

        self.ui = UIPublisher()
        self.ui.init()
        self.ui.publish_state(page="status", status="detecting")

        self.stop_on_cane = False
        self.CONFIDENCE_THRESHOLD = 0.7
        self.cv_image = None

        # BVI target state
        self.bvi_target_box = None   # (x1, y1, x2, y2)
        self.follow_started = False

        self.dist_threshold_m = rospy.get_param("~target_distance_threshold_m", 3.0)  # only lock target if <= 2m
        self.depth_topic_enabled = rospy.get_param("~use_depth", True)

        self.depth_image = None
        self.depth_encoding = None

        # Save directory
        self.save_dir = "/home/mustar/catkin_ws/src/bvi_fyp/src/image_detected"
        os.makedirs(self.save_dir, exist_ok=True)

        # Load models
        self.model_cane = ultralytics.YOLO('/home/mustar/catkin_ws/src/bvi_fyp/src/model/cane_best.pt')
        self.model_person = ultralytics.YOLO('yolov8n.pt')  # COCO model
        self.bridge = CvBridge()

        # Speech services
        rospy.loginfo("Waiting for /speech_to_text, /text_to_speech")
        rospy.wait_for_service("/speech_to_text")
        rospy.wait_for_service("/text_to_speech")

        self.sr_client = rospy.ServiceProxy("/speech_to_text", stt)
        self.tts_client = rospy.ServiceProxy("/text_to_speech", tts)

        rospy.loginfo("Speech services ready.")
        rospy.loginfo("Continuous cane detection node started.")

        # Intro message
        self.speak(
            "Hello. I am your faculty navigation guide robot. "
            "I am now scanning for blind or visually impaired individuals with a guide cane. "
        )

        # ---- Threading: keep only latest frame ----
        self.frame_lock = threading.Lock()
        self.latest_color_msg = None
        self.latest_color_stamp = None
        self.new_frame_event = threading.Event()
        

        # optional: avoid processing too fast
        self.min_process_dt = rospy.get_param("~min_process_dt", 0.0)  # seconds (0 = as fast as possible)

        self.worker_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.worker_thread.start()

        # Image subscriber
        image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)
        if self.depth_topic_enabled:
            depth_topic = rospy.get_param('~depth_topic', '/camera/depth/image_raw')
            self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback, queue_size=1)
            rospy.loginfo(f"Subscribed to depth topic: {depth_topic}")


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

    # Calculate target user distance
    def depth_callback(self, msg_depth):
        try:
            # "passthrough" keeps original encoding (16UC1 or 32FC1)
            depth = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding="passthrough")
            depth = np.flip(depth, axis=1)
            self.depth_image = depth
            self.depth_encoding = msg_depth.encoding  # e.g. "16UC1" or "32FC1"
        except CvBridgeError as e:
            rospy.logwarn(str(e))
            self.depth_image = None

    def estimate_bbox_distance_m(self, bbox):
        """
        Return distance in meters using MEDIAN depth inside the bbox ROI.
        This matches your FollowPerson approach (median over bbox area).
        Returns None if invalid.
        """
        if self.depth_image is None:
            return None

        x1, y1, x2, y2 = map(int, bbox)
        h, w = self.depth_image.shape[:2]

        # clamp bbox to image bounds
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w,     x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h,     y2))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = self.depth_image[y1:y2, x1:x2].astype(np.float32)

        # remove invalid depth
        roi = roi[np.isfinite(roi)]
        roi = roi[roi > 0]

        if roi.size == 0:
            return None

        d = float(np.median(roi)) 

        # unit conversion
        if self.depth_encoding and "16UC1" in self.depth_encoding:
            d = d / 1000.0  # mm -> m
        # if 32FC1, it's usually already meters

        return d
    
    def image_callback(self, msg_color):
        if self.stop_on_cane:
            return

        # store only the latest frame (drop older ones)
        with self.frame_lock:
            self.latest_color_msg = msg_color
            self.latest_color_stamp = msg_color.header.stamp if msg_color.header else rospy.Time.now()

        self.new_frame_event.set()

    def process_loop(self):
        last_t = time.time()

        while not rospy.is_shutdown():
            # wait until a new frame arrives
            self.new_frame_event.wait(timeout=0.5)
            if rospy.is_shutdown():
                break

            # throttle if needed
            if self.min_process_dt > 0:
                now = time.time()
                if now - last_t < self.min_process_dt:
                    time.sleep(max(0.0, self.min_process_dt - (now - last_t)))
                last_t = time.time()

            # grab the latest frame message and clear event
            with self.frame_lock:
                msg_color = self.latest_color_msg
                self.latest_color_msg = None
                self.new_frame_event.clear()

            if msg_color is None:
                continue

            self.process_one_frame(msg_color)

    def process_one_frame(self, msg_color):
        if self.stop_on_cane:
            return

        # Convert image (heavy work starts here, NOT in callback)
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

        results_person = self.model_person(self.cv_image, classes=[0])
        person_boxes = results_person[0].boxes

        persons = []
        canes = []

        # Person detection with IDs
        for pid, box in enumerate(person_boxes):
            conf = box.conf.item()
            cls_id = int(box.cls.item())

            if conf >= self.CONFIDENCE_THRESHOLD and cls_id == 0:
                xyxy = box.xyxy.numpy().flatten()
                x1, y1, x2, y2 = map(float, xyxy)
                persons.append((pid, x1, y1, x2, y2))

                cv2.rectangle(self.cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(
                    self.cv_image,
                    f"id {pid} {conf:.2f}",
                    (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        # Cane detection
        for box in cane_boxes:
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            class_name = results_cane[0].names[cls_id].lower()

            if conf >= self.CONFIDENCE_THRESHOLD and class_name == "guide cane":
                xyxy = box.xyxy.numpy().flatten()
                x1, y1, x2, y2 = map(float, xyxy)
                canes.append((x1, y1, x2, y2))

                cv2.rectangle(self.cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                cv2.putText(self.cv_image, f"cane {conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # Person cane intersection - choose person with highest overlap
        bvi_person_box = None
        bvi_person_id = None
        best_overlap = 0.0

        if persons and canes:
            for (pid, px1, py1, px2, py2) in persons:
                for (cx1, cy1, cx2, cy2) in canes:
                    cane_cx = 0.5 * (cx1 + cx2)
                    cane_cy = 0.5 * (cy1 + cy2)

                    if not (px1 <= cane_cx <= px2) or cane_cy < py1:
                        continue

                    ix1 = max(px1, cx1)
                    iy1 = max(py1, cy1)
                    ix2 = min(px2, cx2)
                    iy2 = min(py2, cy2)

                    iw = max(0.0, ix2 - ix1)
                    ih = max(0.0, iy2 - iy1)
                    overlap = iw * ih

                    if overlap > best_overlap:
                        best_overlap = overlap
                        bvi_person_box = (px1, py1, px2, py2)
                        bvi_person_id = pid

        person_with_cane = False
        if bvi_person_box is not None:
            dist_m = self.estimate_bbox_distance_m(bvi_person_box)

            if dist_m is None:
                rospy.logwarn("Depth unavailable/invalid -> ignoring candidate.")
            elif dist_m <= self.dist_threshold_m:
                person_with_cane = True
                rospy.loginfo(f"BVI within range: {dist_m:.2f}m (<= {self.dist_threshold_m:.2f}m)")
            else:
                rospy.loginfo(f"BVI detected but too far: {dist_m:.2f}m > {self.dist_threshold_m:.2f}m")

        # Action once
        if person_with_cane and not self.follow_started:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{self.save_dir}/person_cane_{timestamp}.jpg"
            tmp_path = "/home/mustar/catkin_ws/src/bvi_fyp/src/tmp/target_user.jpg"
            cv2.imwrite(save_path, self.cv_image)
            cv2.imwrite(tmp_path, self.cv_image)
            rospy.loginfo(f"Saved image at {save_path}")

            self.ui.publish_state(
                page="detection",
                status="target_detected",
                image_path=tmp_path
            )
            rospy.loginfo("PERSON WITH CANE DETECTED! Saving and starting follow-me logic...")

            # cv2.imshow("Cane Detection", self.cv_image)
            # cv2.waitKey(1000)

            self.speak(
                "Excuse me. I have detected a person with a guide cane. Hello, I am your faculty guide robot. "
                "Can you please stop walking and stand still? I will approach you now."
            )

            self.ui.publish_state(page="status", status="approaching")

            self.bvi_target_box = bvi_person_box
            self.start_follow_target(self.bvi_target_box)

            x1, y1, x2, y2 = self.bvi_target_box
            print("BVI person ID:", bvi_person_id)
            print("target bvi individual coordinate: ", x1, y1, x2, y2)

            self.stop_on_cane = True
            return



if __name__ == "__main__":
    try:
        CaneDetection()
        rospy.spin()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down continuous detection node.")
