#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import PointStamped, PoseStamped
import tf2_ros
import tf2_geometry_msgs
from bvi_fyp.srv import BVIBoundingBox, BVIBoundingBoxResponse
from bvi_fyp.srv import BVIPointStamped, BVIPointStampedRequest


class HandleBoundingBox:
    def __init__(self):
        rospy.init_node("handle_bbox_node")

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # need to change BVIBoundingBox service file, track_id --> distance
        # Subscribe to bvi target bounding box and estimated distance
        self.bvi_target_box = None  # (x1, y1, x2, y2)
        self.distance_m = None
        self.bvi_target_received = False
        rospy.loginfo("Here")

        rospy.wait_for_service("/approach_to_user")
        rospy.loginfo("Here")



        self.bbox_sub = rospy.Service('bvi_target_bbox', BVIBoundingBox, self.update_target)
        self.nav_client = rospy.ServiceProxy('approach_to_user', BVIPointStamped)


        # Distance to stop before the user
        self.stop_distance = 1.0

        rospy.loginfo("Handle bbox node running. Waiting for bbox...")


        # Example loop (replace with service call or subscriber)
        # rospy.Timer(rospy.Duration(1.0), self.update_target)

    def update_target(self, msg):
        # --- Replace with actual service call ---
        # Example bounding box (x1, y1, x2, y2)
        self.bvi_target_box = (msg.x1, msg.y1, msg.x2, msg.y2)
        self.distance_m = msg.distance
        self.bvi_target_received = True
        rospy.loginfo("bbox received")
        

        # Convert to PointStamped
        point_robot = self.bbox_to_point(self.bvi_target_box, self.distance_m)

        # Transform to map, optional
        point_map = self.transform_to_map(point_robot)
        if point_map is None:
            return False

        rospy.loginfo(point_map)

        # Send to navigation module
        try:
            self.nav_client(BVIPointStampedRequest(target=point_map))
            rospy.loginfo("pointstamped sent")
            return True

        except Exception as e:
            rospy.logerr(f"Failed sending PointStamped: {e}")
            return False

    def bbox_to_point(self, bbox, distance_m):
        """
        Convert bbox center + distance to PointStamped in robot frame
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2

        # Camera parameters
        image_width = 640
        fov_x = 60.0 * math.pi / 180

        # Fractional offset from center
        frac = (cx - image_width / 2) / (image_width / 2)
        y_robot = distance_m * math.tan(frac * fov_x / 2)
        x_robot = max(0.0, distance_m - self.stop_distance)
        z_robot = 0.0

        point = PointStamped()
        point.header.stamp = rospy.Time.now()
        point.header.frame_id = "base_link"
        point.point.x = x_robot
        point.point.y = y_robot
        point.point.z = z_robot
        return point

    def transform_to_map(self, point):
        try:
            point_map = self.tf_buffer.transform(point, "map", rospy.Duration(1.0))
            return point_map
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None

if __name__ == "__main__":
    try:
        HandleBoundingBox()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down approach user node.")
