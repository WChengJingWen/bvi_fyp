#! /usr/bin/env python3

import rospy
from std_msgs.msg import String, Bool
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
import tf.transformations



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

        # Publisher for beep sound
        self.beep_pub = rospy.Publisher("/beep_control", Bool, queue_size=1)

        # Publisher for goal visualization
        self.goal_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        # Subscribe to RViz initial pose topic to set robot's starting location
        initial_pose = PoseWithCovarianceStamped()
        rospy.Subscriber('initialpose', PoseWithCovarianceStamped,
                         self.update_initial_pose)
        
        # Subscribe to Global Planner # rostopic list, look for correct topic name
        self.global_path_sub = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.global_path_callback)

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

        # --------------------------------------------------------------------------
        # Start a ROS service called 'navigate' to receive navigation requests
        self.service = rospy.Service('navigate', Navigate, self.nav_to_point)
        self.tts_client = rospy.ServiceProxy("text_to_speech", tts)


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
                    rospy.loginfo(f"Loaded {len(self.checkpoints)} checkpoints")
                    # self.publish_status(f"Loaded {len(self.checkpoints)} checkpoints")
                    
                    # Then broadcast each checkpoint individually
                    rospy.sleep(0.5)  # Give subscribers time to connect
                    for name, data in self.checkpoints.items():
                        rospy.loginfo(f"checkpoint:{name}:{json.dumps(data)}")
                        # self.publish_status(f"checkpoint:{name}:{json.dumps(data)}")
                        rospy.sleep(0.1)  # Small delay between messages to ensure delivery
                    
                    return valid_checkpoints

                else:
                    self.checkpoints = {}
                    rospy.loginfo(f"Starting with empty checkpoint list")

                    # self.publish_status("Starting with empty checkpoint list")
            except Exception as e:
                rospy.logerr(f"Error loading checkpoints: {e}")
                self.checkpoints = {}
                # self.publish_status(f"Error loading checkpoints: {e}")

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
        if not self.is_navigating:
            return  # ignore paths if not navigating

        # Need at least 3 points to detect turning
        if len(msg.poses) < 3:
            rospy.loginfo("no poses")
            return

        p1 = msg.poses[0].pose.position
        p2 = msg.poses[1].pose.position
        p3 = msg.poses[2].pose.position

        v1x = p2.x - p1.x
        v1y = p2.y - p1.y
        v2x = p3.x - p1.x
        v2y = p3.y - p1.y

        angle1 = math.atan2(v1y, v1x)
        angle2 = math.atan2(v2y, v2x)
        delta = self.normalize_angle(angle2 - angle1)

        rospy.loginfo(delta)


        # Classification
        if delta > math.radians(20):
            motion = "left"
        elif delta < -math.radians(20):
            motion = "right"
        else:
            return  # ignore small noisy changes
    
        self.publish_motion_audio(motion)

    def publish_motion_audio(self, motion):
        # self.beep_pub.publish(False)
        # rospy.sleep(0.2)

        if motion == "left":
            rospy.loginfo(f"turn left")
            self.speak("Turning left")
        elif motion == "right":
            rospy.loginfo(f"turn right")
            self.speak("Turning right")

        # self.beep_pub.publish(True)


    def speak(self, text):
        try:
            self.tts_client(ttsRequest(text=text))
        except Exception as e:
            rospy.logerr(f"TTS call failed: {e}")

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
    
    def nav_to_target_user(self, point_map):
            
        if self.is_navigating:
            rospy.logwarn("Already navigating. Cancel current goal first.")
            return False
            
        # Create PoseStamped from PointStamped
        pose = PoseStamped()
        pose.header = point_map.header
        pose.pose.position = point_map.point
        
        # Set orientation (facing the target)
        try:
            current_pose = self.get_current_pose()
            if current_pose:
                dx = pose.pose.position.x - current_pose.pose.position.x
                dy = pose.pose.position.y - current_pose.pose.position.y
                yaw = math.atan2(dy, dx)
                quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
                pose.pose.orientation.x = quaternion[0]
                pose.pose.orientation.y = quaternion[1]
                pose.pose.orientation.z = quaternion[2]
                pose.pose.orientation.w = quaternion[3]
            else:
                pose.pose.orientation.w = 1.0
        except Exception as e:
            rospy.logwarn(f"Could not calculate orientation: {e}")
            pose.pose.orientation.w = 1.0
            
        # Create and send goal
        goal = MoveBaseGoal()
        goal.target_pose = pose
        
        try:
            rospy.loginfo("Navigating to detected target")
            # self.publish_status("navigating to detected target")
            self.is_navigating = True
            
            # Publish for visualization
            self.goal_publisher.publish(pose)
            
            # Send goal to move_base
            self.move_base.send_goal(
                goal,
                done_cb=self.navigation_done_callback,
                feedback_cb=self.navigation_feedback_callback
            )
            return True
        except Exception as e:
            rospy.logerr(f"Error sending goal to detected target: {e}")
            # self.publish_status("error: failed to send goal to target")
            self.is_navigating = False
            return False
        
    def nav_to_point(self, request):
        """
        Service callback to navigate the robot to the requested location.
        """
        target = request.target_location

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
        self.goal.target_pose = pose

        rospy.loginfo(f"Going to {target}")

        self.move_base.send_goal(self.goal)
        self.is_navigating = True

        if target != "initial_point":
            self.beep_pub.publish(True)


        # Wait up to 300 seconds for the robot to reach the goal
        waiting = self.move_base.wait_for_result(rospy.Duration(300))
        if waiting:
            rospy.loginfo(f"Reached {target}")
            self.is_navigating = False
            self.beep_pub.publish(False)
            return NavigateResponse(reach=True, message="Reached")
        else:
            self.is_navigating = False
            self.beep_pub.publish(False)
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
                self.beep_pub.publish(False)

            except:
                pass
            rospy.loginfo("Navigation cancelled")
            # self.publish_status("navigation cancelled")
            self.is_navigating = False

    # def publish_status(self, status):
    #     """Publish navigation status"""
    #     msg = String()
    #     msg.data = status
    #     self.status_pub.publish(msg)

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
