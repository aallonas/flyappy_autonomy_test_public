from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
from flyappy_autonomy_code.flyappy_control import FlyappyControl
from flyappy_autonomy_code.flyappy_perception import FlyappyPerception
import numpy as np
from numpy import typing as npt
from typing import Optional

class FlyappyRos:
    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()

        # Rate parameters for timers 
        control_rate = node.declare_parameter("control_rate", 10.0).value
        perception_rate = node.declare_parameter("Perception_rate", 20.0).value
        perception_period = 1.0 / perception_rate
        control_period = 1.0 / control_rate

        self._perception = FlyappyPerception()
        self._control = FlyappyControl()

        self._current_velocity: npt.NDArray[np.float64] = np.array([0.0, 0.0])
        self._current_laser_scan: Optional[npt.NDArray[np.float32]] = None

        # Publisher for sending acceleration commands to Flyappy
        self._pub_acc_cmd = node.create_publisher(
            Vector3,
            "/flyappy_acc",
            1
        )

        # Subscribers to topics from Flyappy game
        self._sub_vel = node.create_subscription(
            Vector3,
            "/flyappy_vel",
            self.velocity_callback,
            10
        )
        self._sub_laser_scan = node.create_subscription(
            LaserScan,
            "/flyappy_laser_scan",
            self.laser_scan_callback,
            10
        )
        self._sub_game_ended = node.create_subscription(
            Bool,
            "/flyappy_game_ended",
            self.game_ended_callback,
            5
        )

        # Timers for control and perception loops
        self.control_timer = node.create_timer(
            control_period,
            self.control_callback
        )
        self.perception_timer = node.create_timer(
            perception_period,
            self.perception_callback
        )

        self._logger.info(
            f"FlyappyRos initialized with {control_rate}Hz control loop and {perception_rate}Hz perception loop"
        )

    def velocity_callback(self, msg: Vector3) -> None:
        # Store current velocities
        self._current_velocity = np.array([msg.x, msg.y], dtype=np.float64)
        # Logging velocities for debug
        self._logger.info(
            f"Velocity: [{msg.x},{msg.y},{msg.z}", 
            throttle_duration_sec=1
            )

    def laser_scan_callback(self, msg: LaserScan) -> None:
        # Store current laser scan
        self._current_laser_scan = np.array(msg.ranges, dtype=np.float32)
        
        # Logging laser angle and range for debug
        self._logger.info(
            f"Laser range: {msg.ranges[0]}, angle: {msg.angle_min}", 
            throttle_duration_sec=1
            )

    def game_ended_callback(self, msg: Bool) -> None:
        if msg.data:
            self._logger.info(f"Crash detected, recieved: {msg.data}")
        else:
            self._logger.info(f"End of countdown, recieved: {msg.data}")

    # Perception callback function
    def perception_callback(self) -> None:
        try:
            pass
            #self._perception.perception_loop()

        except Exception as e:
            self._logger.error(f"Perception loop error: {e}")

    # Control callback function
    def control_callback(self) -> None:
        try:
            pass
            #self._control.control_loop()

        except Exception as e:
            self._logger.error(f"Control loop error: {e}")
