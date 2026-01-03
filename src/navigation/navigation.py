#! /usr/bin/env python3

import rospy
from std_msgs.msg import String
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Twist, Pose, PoseWithCovarianceStamped, Point, Quaternion, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
from bvi_fyp.srv import Navigate, NavigateResponse
import json
import os
from nav_msgs.msg import Path
import math
from bvi_fyp.srv import tts, ttsRequest
from threading import Lock



# Global variables for storing initial pose only once
original = 0
start = 0


class NavToPoint:
    def __init__(self):
        # Ensure cleanup function is called on shutdown
        rospy.on_shutdown(self.cleanup)

        # Create an action client to interact with the move_base action server
        self.move_base = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction)

        rospy.loginfo("Waiting for move_base action server...")

        # Wait for the action server to become available
        self.move_base.wait_for_server(rospy.Duration(120))
        rospy.loginfo("Connected to move base server.")

        # Subscribe to RViz initial pose topic to set robot's starting location
        initial_pose = PoseWithCovarianceStamped()
        rospy.Subscriber('initialpose', PoseWithCovarianceStamped,
                         self.update_initial_pose)
        
        # Subscribe to Global Planner # rostopic list, look for correct topic name
        self.global_path_sub = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, self.global_path_callback)

        # Publisher for direct velocity commands (used for in-place rotation)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.is_navigating = False
        self.last_motion_state = None  # straight / left / right


        rospy.loginfo(
            "*** Click the 2D Pose Estimate button in RViz to set the robot's initial pose...")

        # Wait until user sets the initial pose in RViz
        rospy.wait_for_message('initialpose', PoseWithCovarianceStamped)

        # Wait until the timestamp is set, indicating a valid pose
        while initial_pose.header.stamp == "":
            rospy.sleep(1)

        rospy.loginfo("Starting navigation node...")
        rospy.sleep(1)

        # Get initial location
        # quaternion = quaternion_from_euler(0.0, 0.0, 0.0)
        # self.origin = Pose(Point(0, 0, 0), Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3]))
        # --------------------------------------------------------------------------

        # Predefined named locations with their corresponding coordinates and orientations
        '''
        How to get location?
        ## Run the navigation with your map first
        ## Control the robot to the locations that you wanna save
        ## Run `rostopic echo /amcl_pose` to get the coodinate x, y and orientation z, w
        ## Update the locations
        '''

        # Thread safety
        self.file_lock = Lock()

        # File path for checkpoint storage
        self.checkpoint_file = os.path.join(os.path.dirname(__file__), 'checkpoints.json')
        
        # Initialize checkpoints
        self.checkpoints = self.load_checkpoints()

        # self.locations = {
        #     'shelf': Pose(Point(1.3398, 0.6447, 0), Quaternion(0, 0, 0.5642, 0.8256)),
        #     'front door': Pose(Point(-2.725, -0.4970, 0), Quaternion(0, 0, -0.6906, 0.7232)),
        #     'backdoor': Pose(Point(-2.2345, 4.2042, 0), Quaternion(0, 0, 0.9504, 0.3107))
        
            # 'bag_location': Pose(Point(0.39160446336239296, -0.5161442605907655, 0), Quaternion(0, 0, 0.01574806059039607, 0.9998759916047796)),
            # 'start_point': Pose(Point(0.24595393153346543, 0.0490977198374147, 0), Quaternion(0, 0, 0.800840237438781, 0.5988780460987002)),
            # 'living_room': Pose(Point(0.17685625779139647, 0.47992523468894727, 0), Quaternion(0, 0, 0.7532693005546219, 0.6577122173427756)),
            # 'bedroom': Pose(Point(0.20386362237733346, -0.05311959611005832, 0), Quaternion(0, 0, -0.704544922359989, 0.7096593918047989)),
            # 'kitchen': Pose(Point(1.530233187266717, 0.8213574583047445, 0), Quaternion(0, 0, 0.6731516763193642, 0.7395044426292718)),
            # 'study_room': Pose(Point(0.20386362237733346, -0.05311959611005832, 0), Quaternion(0, 0, -0.704544922359989, 0.7096593918047989)),
            # 'entrance': Pose(Point(0.20386362237733346, -0.05311959611005832, 0), Quaternion(0, 0, -0.704544922359989, 0.7096593918047989)),
            # 'trash_bin': Pose(Point(0.20386362237733346, -0.05311959611005832, 0), Quaternion(0, 0, -0.704544922359989, 0.7096593918047989)),

            # 'maker_chair': Pose(Point(4.806, 0.4116, 0), Quaternion(0, 0, -0.1347, 0.9908)),
            # 'maker_table': Pose(Point(4.4608, -2.0033, 0), Quaternion(0, 0, -0.7158, 0.6982)),
            # 'kitchen': Pose(Point(4.7412, -0.54295, 0), Quaternion(0, 0,-0.3371, 0.9414)),
            # 'Shelf1': Pose(Point(4.1618, -1.5304, 0), Quaternion(0, 0, 0.9976, 0.2170)),
            # 'Shelf2': Pose(Point(4.1781, -1.1627, 0), Quaternion(0, 0, -0.9992,  0.03939)),
            # 'Shelf3': Pose(Point(4.24252, -0.8534, 0), Quaternion(0, 0,  -0.9984, 0.05482)),
            # 'RedTable': Pose(Point(5.2189, -1.2945, 0), Quaternion(0, 0, -0.6971, 0.7169)),
            # 'table1': Pose(Point(5.0133, -0.4718, 0), Quaternion(0, 0, 0.6557, 0.7549)),
            # 'table2': Pose(Point(5.7666, -0.5163, 0), Quaternion(0, 0, 0.6296, 0.7768)),
            # 'halfway': Pose(Point(2.9546, 0.8705, 0), Quaternion(0, 0, -0.7496, 0.6618)),
            # 'entrance recpt': Pose(Point(0.5808, 1.1929, 0), Quaternion(0, 0, 0.99790, 0.06462)),
            # 'seat_find': Pose(Point(1.1175, -0.1197, 0), Quaternion(0, 0, -0.5155, 0.8568)),
            # 'direct_sofa': Pose(Point(2.2832, -0.1862, 0), Quaternion(0, 0, -0.6013, 0.7989)),
            # 'blue_seat': Pose(Point(0.8119, -1.650, 0), Quaternion(0, 0, -0.5293, 0.84837))
        # }

        # --------------------------------------------------------------------------
        # Start a ROS service called 'navigate' to receive navigation requests
        self.service = rospy.Service('navigate', Navigate, self.nav_to_point)
        self.tts_client = rospy.ServiceProxy("/text_to_speech", tts)


    def load_checkpoints(self):
        """Load checkpoints from file with thread safety"""
        with self.file_lock:
            try:
                if os.path.exists(self.checkpoint_file):
                    with open(self.checkpoint_file, 'r') as f:
                        new_checkpoints = json.load(f)
                    
                    # Validate all checkpoints in the file
                    valid_checkpoints = {}
                    for name, data in new_checkpoints.items():
                        if self.validate_checkpoint(data):
                            valid_checkpoints[name] = data
                    
                    self.checkpoints = valid_checkpoints
                    
                    # First notify about the total number of checkpoints
                    self.publish_status(f"Loaded {len(self.checkpoints)} checkpoints")
                    
                    # Then broadcast each checkpoint individually
                    rospy.sleep(0.5)  # Give subscribers time to connect
                    for name, data in self.checkpoints.items():
                        self.publish_status(f"checkpoint:{name}:{json.dumps(data)}")
                        rospy.sleep(0.1)  # Small delay between messages to ensure delivery
                else:
                    self.checkpoints = {}
                    self.publish_status("Starting with empty checkpoint list")
            except Exception as e:
                rospy.logerr(f"Error loading checkpoints: {e}")
                self.checkpoints = {}
                self.publish_status(f"Error loading checkpoints: {e}")

    def validate_checkpoint(self, checkpoint_data):
        """Validate checkpoint data structure"""
        required_fields = {
            'position': ['x', 'y', 'z'],
            'orientation': ['x', 'y', 'z', 'w']
        }
        
        try:
            for field, subfields in required_fields.items():
                if field not in checkpoint_data:
                    return False
                for subfield in subfields:
                    if subfield not in checkpoint_data[field]:
                        return False
                    if not isinstance(checkpoint_data[field][subfield], (int, float)):
                        return False
            return True
        except Exception:
            return False

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def global_path_callback(self, msg: Path):
        # Need at least 3 points to detect turning
        if len(msg.poses) < 3:
            return

        p1 = msg.poses[0].pose.position
        p2 = msg.poses[1].pose.position
        p3 = msg.poses[2].pose.position

        v1x = p2.x - p1.x
        v1y = p2.y - p1.y
        v2x = p3.x - p2.x
        v2y = p3.y - p2.y

        angle1 = math.atan2(v1y, v1x)
        angle2 = math.atan2(v2y, v2x)
        delta = self.normalize_angle(angle2 - angle1)

        # Classification
        if abs(delta) < math.radians(10):
            motion = "straight"
        elif delta > math.radians(20):
            motion = "left"
        elif delta < -math.radians(20):
            motion = "right"
        else:
            return  # ignore small noisy changes

        # Avoid repeating same audio cue
        if motion != self.last_motion_state:
            self.last_motion_state = motion
            self.publish_motion_audio(motion)

    def publish_motion_audio(self, motion):
        if motion == "straight":
            self.tts_client(ttsRequest(text="Go straight"))
        elif motion == "left":
            self.tts_client(ttsRequest(text="Turning left"))
        elif motion == "right":
            self.tts_client(ttsRequest(text="Turning right"))
        
    def create_pose_stamped(self, position, orientation):
        """Create a PoseStamped message"""
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        
        # Set position
        pose.pose.position.x = float(position['x'])
        pose.pose.position.y = float(position['y'])
        pose.pose.position.z = float(position.get('z', 0.0))
        
        # Set orientation
        pose.pose.orientation.x = float(orientation['x'])
        pose.pose.orientation.y = float(orientation['y'])
        pose.pose.orientation.z = float(orientation['z'])
        pose.pose.orientation.w = float(orientation['w'])
        
        return pose
    
    def nav_to_point(self, request):
        """
        Service callback to navigate the robot to the requested location.
        """
        target = request.target_location.strip().lower()

        if target == 'stop':
            self.cancel_navigation()
            return NavigateResponse(reach=False, message="Stopped navigation")

        if self.is_navigating:
            rospy.logwarn("Already navigating. Cancel current goal first.")
            return NavigateResponse(reach=False, message="Already navigating. Cancel current goal first")
            
        if target not in self.checkpoints:
            rospy.logerr(f"Checkpoint {target} not found")
            # self.publish_status(f"error: checkpoint {target} not found")
            return NavigateResponse(reach=False, message="Checkpoint not found")
        
            
        checkpoint = self.checkpoints[target]
        pose = self.create_pose_stamped(
            checkpoint['position'],
            checkpoint['orientation']
        )
        
        self.goal = MoveBaseGoal()
        rospy.loginfo("Ready to go.")

        # Set goal frame and timestamp
        self.goal.target_pose.header.frame_id = 'map'
        self.goal.target_pose.header.stamp = rospy.Time.now()

        # Get destination coordinates from the dictionary based on request
        # coordinate = self.locations[pose.target_location]
        self.goal.target_pose.pose = pose

        rospy.loginfo(f"Going to {pose}")

        # Rotate 180 degrees in-place before starting navigation
        try:
            self.rotate_in_place(math.pi)
        except Exception as e:
            rospy.logwarn(f"Rotation before navigation failed: {e}")

        self.tts_client(ttsRequest(text="Please grab the handle and follow me"))

        # Wait for 5 seconds
        rospy.loginfo("Waiting for 5 seconds, let user grab handle...")
        rospy.sleep(5)

        self.move_base.send_goal(self.goal)
        self.is_navigating = True

        # Wait up to 300 seconds for the robot to reach the goal
        waiting = self.move_base.wait_for_result(rospy.Duration(300))
        if waiting:
            rospy.loginfo(f"Reached {pose}")
            self.is_navigating = False
            return NavigateResponse(reach=True, message="Reached")
        else:
            self.is_navigating = False
            return NavigateResponse(reach=False, message="Failed to reach point")
    
    def cancel_navigation(self):
        """Cancel current navigation goal"""
        if self.is_navigating:
            # Cancel all goals
            self.move_base.cancel_all_goals()
            # Wait a bit to ensure the cancellation is processed
            rospy.sleep(0.5)
            # Force stop by sending an empty goal at current position
            try:
                current_pose = PoseStamped()
                current_pose.header.frame_id = "map"
                current_pose.header.stamp = rospy.Time.now()
                # Send empty goal to stop movement
                goal = MoveBaseGoal()
                goal.target_pose = current_pose
                self.move_base.send_goal(goal)
                self.move_base.cancel_all_goals()
            except:
                pass
            rospy.loginfo("Navigation cancelled")
            self.publish_status("navigation cancelled")
            self.is_navigating = False

        def rotate_in_place(self, angle, angular_speed=0.2):
            """Rotate the robot in-place by `angle` radians at `angular_speed` (rad/s).
            Positive angle = counter-clockwise. This uses `/cmd_vel` to command rotation.
            """
            # Normalize angle to [-pi, pi]
            angle = self.normalize_angle(angle)

            # Determine rotation direction
            direction = 1.0 if angle >= 0 else -1.0
            target_angle = abs(angle)

            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0

            rate = rospy.Rate(10)
            start_time = rospy.Time.now()
            duration = rospy.Duration(target_angle / angular_speed)

            twist.angular.z = direction * abs(angular_speed)

            # Publish until desired rotation time elapsed
            while rospy.Time.now() - start_time < duration and not rospy.is_shutdown():
                self.cmd_vel_pub.publish(twist)
                rate.sleep()

            # Stop rotation
            twist.angular.z = 0.0
            for _ in range(3):
                self.cmd_vel_pub.publish(twist)
                rate.sleep()

    def update_initial_pose(self, initial_pose):
        """
        Callback function to update the robot's initial pose once.
        """
        self.initial_pose = initial_pose
        global original
        if original == 0:
            # Store initial pose only once
            self.origin = self.initial_pose.pose.pose
            original = 1

    def cleanup(self):
        """
        Called on shutdown. Cancels current move_base goal.
        """
        rospy.loginfo("Shutting down navigation...")
        self.move_base.cancel_goal()


if __name__ == "__main__":
    rospy.init_node('navi_point')
    try:
        NavToPoint()
        rospy.spin()
    except:
        pass
