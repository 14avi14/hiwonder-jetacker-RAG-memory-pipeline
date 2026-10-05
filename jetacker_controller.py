"""jetacker_controller.py - sends sensor data, runs navigation commands

This script runs on the HiWonder JetAcker. It is responsible for
sending image and position data to the server. If it recieves
instructions for a goal position afterwards, then script will publish
that to the `/goal_pose` topic, which should lead to the car
automatically navigating to that position in real-life.

*IMPORTANT*: In order for the `/goal_pose` topic to be visible, SLAM
navigation must be on.
"""


import socket
import time
import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy # Needed to specify QoS
from sensor_msgs.msg import Image 
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped # For controlling the car
from actions_msgs.msg import GoalStatusArray, GoalStatus
from cv_bridge import CvBridge # Convert ROS2 Image to opencv image

# Local imports
import socket_ops

sim_topics = {
    "position": "/odom", # Need to update to account for global positioning
    "cmd_vel": "/controller/cmd_vel",
    "rgb_cam": "/depth_cam/depth_cam",
    "goal_pose": "/goal_pose",
    "goal_status": "/navigate_to_pose/_action/status"
}

real_topics = {
    "position": "/amcl_pose", # Gives pose relative to 'map' frame collected by SLAM
    "cmd_vel": "/controller/cmd_vel",
    "rgb_cam": "/depth_cam/rgb/image_raw",
    "goal_pose": "/goal_pose"
    "goal_status": "/navigate_to_pose/_action/status"
}

class ControlReciever(Node):
    def __init__(self, is_sim=False):
        super().__init__("control_reciever")
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST,
                         depth=10)
        
        if is_sim:
            topics = sim_topics
        else:
            topics = real_topics

        self.goal_pub = self.create_publisher(PoseStamped,
                                              topics["goal_pose"],
                                              qos)
        self.goal_status_sub = self.create_subscription(GoalStatusArray,
                                                        topics["goal_status"],
                                                        self.update_target_status,
                                                        qos)
                
        # Image
        self.rgb_sub = self.create_subscription(Image,
                                                topics["rgb_cam"],
                                                self.set_trajectory,
                                                qos)

        # Position relative to 'map'/allocentric frame
        self.global_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                        topics["position"],
                                                        self.update_car_state,
                                                        qos)

        self.bridge = CvBridge() # ROS2 Image to cv2

        self.car_state = None
        self.goal_pose = None

        self.connection = socket.socket()
        self.connection.connect(socket_ops.SERVER_ADDRESS)

    def update_car_state(self, raw_pose_data):
        self.get_logger().info("GLOBAL POSITION CHANGED")
        self.get_logger().info(f"Frame ID: {raw_pose_data.header.frame_id}")

        orientation = raw_pose_data.pose.pose.orientation
        angle = quaternions_to_euler_z(orientation.w,
                                       orientation.x,
                                       orientation.y,
                                       orientation.z)

        position = raw_pose_data.pose.pose.position
        self.car_state = {
            "heading_deg": round(np.degrees(angle)),
            "position_xy": [position.x, position.y]
        }

    def update_target_status(self, goal_status_array):
        most_recent = max(goal_status_array.status_list,
                          key=lambda status: status.goal_info.stamp.sec)

        status_num = most_recent.status

        if status_num == GoalStatus.STATUS_UNKNOWN:
            self.get_logger().info("GOAL STATUS UNKNOWN - RESETTING")
            self.goal_pose = None
        elif status_num == GoalStatus.ACCEPTED:
            pass
        elif status_num == GoalStatus.EXECUTING:
            pass
        elif status_num in [GoalStatus.CANCELING, GoalStatus.CANCELED, GoalStatus.ABORTED]:
            self.get_logger().info("GOAL CANCELING - RESETTING")
            self.goal_pose = None
        elif status_num == GoalStatus.SUCCEEDED:
            self.get_logger().info("GOAL ACHIEVED - RESETTING")
            self.goal_pose = None

    def set_trajectory(self, rgb):
        if self.car_state is None:
            return # Wait until global positioning is done
        
        if self.goal_pose is not None:
            return # Wait until goal is either aborted/canceled or reached

        car_state_json = json.dumps(self.car_state)

        cv_bgr_image = self.bridge.imgmsg_to_cv2(rgb)[..., ::-1] # Reverse index changes RGB to BGR
        
        self.get_logger().info("STARTING TO SEND DATA")
        socket_ops.send_image(self.connection, cv_bgr_image)
        socket_ops.send_string(self.connection, car_state_json)
        socket_ops.send_string(self.connection, str(time.time()))
        self.get_logger().info("MESSAGE SENT")

        result = json.loads(socket_ops.recieve_string(self.connection))
        if result.get("goal_pose") is None:
            self.goal_pose = None
            self.get_logger().info("[MESSAGE RECIVED] No target")
            self.get_logger().info(f"REASONING: {result['reasoning']}")
        else:
            self.get_logger().info(f"[MESSAGE RECIVED] Target: {result['goal_pose']}")

            self.goal_pose = {
                "heading_deg": result["goal_pose"][2],
                "position_xy": result["goal_pose"][:2]
            }

            msg = PoseStamped()

            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()

            msg.pose.position.x = float(result["goal_pose"][0])
            msg.pose.position.y = float(result["goal_pose"][1])
            w, x, y, z = euler_to_quaternions(0, 0, np.radians(result["goal_pose"][2]))
            msg.pose.orientation.w = 1.0
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            
            self.goal_pub.publish(msg)


def euler_to_quaternions(roll, pitch, yaw):
    cr = np.cos(roll/2)
    cp = np.cos(pitch/2)
    cy = np.cos(yaw/2)

    sr = np.sin(roll/2)
    sp = np.sin(pitch/2)
    sy = np.sin(yaw/2)

    w = cr * cp * cy + sr * sp * sy
    x = cr * sp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return w, x, y, z

def quaternions_to_euler_z(w, x, y, z):
    return np.arctan2(2*(w*z + x*y), 1-2*(y**2 + z**2))
                

def main(args=None):
    rclpy.init(args=args)
    node = ControlReciever(is_sim=False)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
        
        
