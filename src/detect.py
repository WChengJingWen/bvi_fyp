#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import ultralytics
from datetime import datetime
import os

class GroceryDetection:
    def __init__(self):
        rospy.init_node('cane_detection')

        # Load YOLO models
        self.model_cane = ultralytics.YOLO('/home/mustar/catkin_ws/src/bvi_fyp/src/model/cane_best.pt')
        self.model_person = ultralytics.YOLO('yolov8n.pt')  # COCO model

        self.bridge = CvBridge()
        image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)

        # Stop flag
        self.stop_on_cane = False

        self.CONFIDENCE_THRESHOLD = 0.7
        self.cv_image = None

        # Save directory
        self.save_dir = "/home/mustar/catkin_ws/src/bvi_fyp/src/image_detected"
        os.makedirs(self.save_dir, exist_ok=True)

        rospy.loginfo("Continuous cane detection node started.")


    def image_callback(self, msg_color):

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

        if persons and canes:
            for (px1, py1, px2, py2) in persons:
                for (cx1, cy1, cx2, cy2) in canes:
                    cane_cx = 0.5 * (cx1 + cx2)
                    cane_cy = 0.5 * (cy1 + cy2)

                    # cane must be horizontally inside person body
                    if (px1 <= cane_cx <= px2) and (cane_cy >= py1):
                        person_with_cane = True
                        break
                if person_with_cane:
                    break

        # -------------------------
        # SAVE & STOP
        # -------------------------
        if person_with_cane:
            rospy.loginfo("PERSON WITH CANE DETECTED! Saving and stopping...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{self.save_dir}/person_cane_{timestamp}.jpg"
            cv2.imwrite(save_path, self.cv_image)

            rospy.loginfo(f"Saved image at {save_path}")

            self.stop_on_cane = True
            return

        # Show detection
        cv2.imshow("Cane Detection", self.cv_image)
        cv2.waitKey(1)



if __name__ == "__main__":
    try:
        GroceryDetection()
        rospy.spin()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down continuous detection node.")
