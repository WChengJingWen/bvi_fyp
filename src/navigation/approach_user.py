#!/usr/bin/env python

# from robocup_home.srv import FollowMe, FollowMeResponse
import rospy
from geometry_msgs.msg import Twist
import json
from ultralytics import YOLO
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from concurrent.futures import ThreadPoolExecutor
from bvi_fyp.srv import BVIBoundingBox, BVIBoundingBoxResponse

# max speed and turn

MAX_SPEED = 0.6
MAX_TURN = 1

MAX_LIN_VEL = 0.26
MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.1


class ApproachUser:
    def __init__(self) -> None:
        rospy.init_node('approach_user')

        # yolo detect params
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.track = False
        self.model = YOLO("/home/mustar/yolov8n.pt")
        self.detect_result = {}

        # follow target flag
        self.target_flag = False
        self.new_frame_ready = True
        # follow obj -> need to change back to false
        self.follow = False
        self.follow_obj_id = -1

        # moving
        self.move_cmd = Twist()
        self.control_speed = 0
        self.control_turn = 0
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # Called in Server
        self.sub_color = Subscriber('/camera/color/image_raw', Image)
        self.sub_depth = Subscriber('/camera/depth/image_raw', Image)
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.images_callback)

        # Subscribe
        # self.service = rospy.Service('follow_me', FollowMe, self.set_follow_state)
        
        # Subscribe to bvi target bounding box
        self.bvi_target_box = None  # (x1, y1, x2, y2)
        self.bvi_target_received = False
        # self.bbox_sub = rospy.Service('bvi_target_bbox', BVIBoundingBox, self.bbox_callback)


        # change follow state according to node
        # self.choose_person = rospy.Subscriber(
        #     'start_follow', Bool, self.set_follow_state)

    def bbox_callback(self, msg):
        # Receive bounding box from teammate.
        # BVIBoundingBox should contain x1, y1, x2, y2 (image coordinates).
        
        self.bvi_target_box = (msg.x1, msg.y1, msg.x2, msg.y2)
        self.bvi_target_received = True
        self.follow = True  # Start following upon receiving target bbox
        rospy.loginfo(f"Received BVI target bbox: {self.bvi_target_box}")
        return BVIBoundingBoxResponse(success=True)

    # def set_follow_state(self, req):
    #     if req.state:  # If the request is to start following
    #         print("Start following")
    #         self.follow = True
    #         self.target_flag = False # Reset target_flag when starting a new follow
    #         self.follow_obj_id = -1 # Reset follow_obj_id

    #         response = FollowMeResponse(
    #             follow_msg="Started following"
    #         )
    #         return response
    #     else: # If the request is to stop following
    #         print("Stop following")
    #         self.follow = False
    #         # Immediately stop the robot by publishing a zero Twist message
    #         self.control_speed = 0
    #         self.control_turn = 0
    #         self.move()
    #         return FollowMeResponse(follow_msg="Stop following")

    def images_callback(self, msg_color, msg_depth):
        try:
            img_color = CvBridge().imgmsg_to_cv2(msg_color, 'bgr8')
            img_depth = CvBridge().imgmsg_to_cv2(msg_depth, 'passthrough')
        except CvBridgeError as e:
            rospy.logwarn(str(e))
            return

        if img_color.shape[0:2] != img_depth.shape[0:2]:
            rospy.logwarn('彩色图像和深度图像分辨率不同')
            return
        
        # Check if follow is true AND if the executor is not currently busy
        # This prevents flooding the executor with new tasks if previous ones are still running
        if self.bvi_target_received:
            if (self.executor._work_queue.empty() and not self.executor._broken):
                frame = np.flip(img_color.copy(), axis=1) # mirrors the image horizontally(left to right)
                depth_flipped = np.flip(img_depth.copy(), axis=1)

                self.executor.submit(self.process_frame_thread, frame, img_color.copy(), img_depth.copy())
                # AI said should be
                # self.executor.submit(self.process_frame_thread, frame, frame.copy(), depth_flipped.copy())

            self.move()


    def process_frame_thread(self, frame, img_color, img_depth):
        # if not self.follow:
        #     rospy.loginfo("Skipping frame processing: self.follow is False.")
        #     self.control_speed = 0
        #     self.control_turn = 0
        #     self.move()
        #     return # Exit early if follow is false
        
        if not self.bvi_target_received:
            rospy.loginfo("Skipping frame processing. Waiting for teammate BVI target bbox...")
            self.control_speed = 0
            self.control_turn = 0
            self.move()
            return
        
        # try:
        results = self.model.track(source=frame, conf=0.5, iou=0.5, classes=[
                                    0], persist=True, tracker="bytetrack.yaml", show=True)
        result = results[0]
        result_json = json.loads(result.tojson())

        if not self.target_flag and self.bvi_target_box is not None:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            frame_x_center = 320 # Assuming 640 width / 2
            min_diff = float('inf')

            target_bbox = self.bvi_target_box
            tx1, ty1, tx2, ty2 = target_bbox
            # tx1 = 640 - tx1  # original frame flipped, target x must also flip
            # tx2 = 640 - tx2
            target_center_x = (tx1 + tx2) / 2
            target_center_y = (ty1 + ty2) / 2

            # Ensure there are boxes and track_ids to prevent errors if no person is detected initially
            if len(boxes) > 0 and len(track_ids) > 0:
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    cx = x
                    cy = y
                    # Calculate Euclidean distance between YOLO-detected bounding box and the BVI target bounding box
                    diff = np.sqrt((cx - target_center_x)**2 + (cy - target_center_y)**2)
                    if diff < min_diff:
                        min_diff = diff
                        self.follow_obj_id = track_id
                self.target_flag = True # Only set to True if a target was identified
            else:
                rospy.loginfo("No matching YOLO person found for BVI target")
                self.control_speed = 0
                self.control_turn = 0
                self.move()
                return # Stop and return if no initial target

        rospy.loginfo(f"target_id: {self.follow_obj_id}")
        x1 = y1 = x2 = y2 = -1
        # Find the coordinates for the tracked object
        found_target_in_frame = False
        for obj in result_json:
            if "track_id" in obj and obj["track_id"] == self.follow_obj_id:
                x1 = obj["box"]["x1"]
                y1 = obj["box"]["y1"]
                x2 = obj["box"]["x2"]
                y2 = obj["box"]["y2"]
                found_target_in_frame = True
                break # Found the target, no need to check other objects

        if not found_target_in_frame: # If the tracked person is lost or not in the current frame
            rospy.loginfo(f"Target (ID: {self.follow_obj_id}) not found in current frame.")
            self.target_flag = False
            self.control_speed = 0
            self.control_turn = 0
            self.move()
            return

        current_depth_img = img_depth[int(y1):int(y2), int(x1):int(x2)]
        if current_depth_img.size == 0 or np.all(current_depth_img == 0): # Check for empty or all zero (invalid) ROI
            rospy.logwarn("Depth ROI is empty or contains only zero values. Cannot calculate depth.")
            self.control_speed = 0
            self.control_turn = 0
            self.move()
            return
        
        current_depth = np.median(current_depth_img)
        print(f"Distance: {int(current_depth/10)}cm")

        # You might want to filter out very large or very small depth values if they are noise
        if current_depth == 0: # Assuming 5000mm is an unrealistic depth
            rospy.logwarn("Invalid or zero depth reading from target ROI. Halting.")
            self.control_speed = 0
            self.control_turn = 0
            self.move()
            return

        # --- Speed Control Logic ---
        depth_threshold = 1000 # Target distance in mm (1.2 meters)
        if current_depth > depth_threshold + 1300: # Far away: > 2.5 meters
            speed = 0.5
        elif current_depth > depth_threshold + 600: # Far: 1.8 - 2.5 meters
            speed = 0.3
        elif current_depth > depth_threshold: # A bit far: 1.8 - 1.3 meters
            speed = 0.1
        elif current_depth < depth_threshold - 600: # Stop when too close: < 0.6 meters
            self.follow = False
            self.control_speed = 0
            self.control_turn = 0
            self.move()
            return
        elif current_depth < depth_threshold: # A bit close: 0.6 - 1 meters
            speed = -0.1
        else: # Within ideal range 1 to 1.3 meters
            rospy.loginfo("Target within ideal range (1-1.3m). Stopping robot.")
            self.follow = False
            self.control_speed = 0
            self.control_turn = 0
            self.move()
            return

        # --- Turn Control Logic ---
        obj_center = (x1 + x2) / 2
        diff_center = 320 - obj_center # Assuming image width 640, center is 320

        if abs(diff_center) <= 30: # Target is centered
            turn = 0
        elif abs(diff_center) > 250: # Far left/right
            turn = 0.5
        elif abs(diff_center) > 160: # Moderate left/right
            turn = 0.3
        elif abs(diff_center) > 60: # Slight left/right
            turn = 0.1
        else: # Very slight adjustment
            turn = 0.05

        if diff_center > 0: # Target is to the left of center, need to turn left (negative angular Z)
            turn = -turn

        # Make sure this check is inside the process_frame_thread *before* publishing move commands
        if not self.follow:
            rospy.loginfo("Self.follow became False during processing. Halting robot.")
            self.control_speed = 0
            self.control_turn = 0
            self.move()
            return # Exit if follow is false
        
        self.control_speed, self.control_turn = self.change_speed(speed, turn)

        # finally:
        #     self.new_frame_ready = True

    # moving function
    def move(self):
        twist = Twist()
        twist.linear.x = self.control_speed
        twist.angular.z = self.control_turn
        self.cmd_vel_pub.publish(twist)

    # change speed function
    def change_speed(self, control_speed, control_turn):
        print('Initial speed:', control_speed, 'target_turn:', control_turn)
        
        target_speed = MAX_SPEED * control_speed 
        target_turn = MAX_TURN * control_turn
        print('Target speed:', target_speed, 'target_turn:', target_turn)

        # This section ensures smooth acceleration/deceleration
        if target_speed > self.control_speed:
            self.control_speed = min(target_speed, self.control_speed + 0.03)
        elif target_speed < self.control_speed:
            self.control_speed = max(target_speed, self.control_speed - 0.05)
        else:
            self.control_speed = target_speed

        if target_turn > self.control_turn:
            self.control_turn = min(target_turn, self.control_turn + 0.1)
        elif target_turn < self.control_turn:
            self.control_turn = max(target_turn, self.control_turn - 0.1)
        else:
            self.control_turn = target_turn

        if self.control_turn > 0.1 or self.control_turn < -0.1:
            self.control_speed = min(self.control_speed, 0.1)  # stop when turning
        print('Changed speed, control_speed:', self.control_speed, 'control_turn:', self.control_turn)


if __name__ == "__main__":
    try:
        node = ApproachUser()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Approach user node terminated.")
