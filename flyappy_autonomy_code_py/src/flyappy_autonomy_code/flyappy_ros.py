from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
from flyappy_autonomy_code.flyappy_control import FlyappyControl
from flyappy_autonomy_code.flyappy_perception import FlyappyPerception
from flyappy_autonomy_code.flyappy_planning import FlyappyPlanning
import numpy as np
from numpy import typing as npt
from typing import Optional
import time

class FlyappyRos:
    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()

        self._current_position: npt.NDArray[np.float64] = np.array([0.0, 1.5])
        self._previous_position: npt.NDArray[np.float64] = np.array([0.0, 1.5])

        self._previous_x_error, self._x_integral = 0.0, 0.0
        self._previous_y_error, self._y_integral = 0.0, 0.0

        self._obstacle_map: npt.NDArray[np.bool_] = np.zeros((408,), dtype=np.bool_)

        self._hole: Optional[tuple[int, int]] = None  
        self._target_pos: Optional[tuple[int, int]] = (0, 1.5)

        # Rate parameters for timers 
        self.control_rate = node.declare_parameter("control_rate", 30.0).value
        self.perception_rate = node.declare_parameter("perception_rate", 30.0).value
        self.planning_rate = node.declare_parameter("planning_rate", 5.0).value
        self.perception_period = 1.0 / self.perception_rate
        self.control_period = 1.0 / self.control_rate
        self.planning_period = 1.0 / self.planning_rate

        self._perception = FlyappyPerception()
        self._control = FlyappyControl()
        self._planning = FlyappyPlanning()

        self._current_velocity: npt.NDArray[np.float64] = np.array([0.0, 0.0])
        self._current_laser_scan: Optional[LaserScan] = None

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

        # Timers for control, perception and planning loops
        self.control_timer = node.create_timer(
            self.control_period,
            self.control_callback
        )
        self.perception_timer = node.create_timer(
            self.perception_period,
            self.perception_callback
        )
        self.planning_timer = node.create_timer(
            self.planning_period,
            self.planning_callback
        )

        self._logger.info(
            f"FlyappyRos initialized with {self.control_rate}Hz control loop and {self.perception_rate}Hz perception loop"
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
        self._current_laser_scan = msg
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
        start_time = time.perf_counter()
        try:
            self._previous_position = self._current_position
            self._current_position = self._perception.compute_position(self._current_velocity,self._previous_position,self.control_period)
            self._logger.info(
                f"Position: [{self._current_position[0]}, {self._current_position[1]}]", 
                throttle_duration_sec=1 )
            
            if self._current_laser_scan is not None:
                obstacles = self._perception.compute_obstacles(self._current_laser_scan)
                self._logger.info(
                    f"Obstacles: {obstacles}", 
                    throttle_duration_sec=1 )
                self._perception.add_obstacles_to_map(self._obstacle_map, obstacles, self._current_position)
                self._logger.info(
                    f"Obstacle map updated{self._obstacle_map}", 
                    throttle_duration_sec=1 )
            
        except Exception as e:
            self._logger.error(f"Perception loop error: {e}")
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._logger.info(f"[TIMING] Perception loop: {elapsed_ms:.2f}ms", throttle_duration_sec=1)
    
    # Control callback function
    def control_callback(self) -> None:

        start_time = time.perf_counter()        
        try:
            # x-axis control
            self._x_acc ,self._previous_x_error, self._x_integral = self._control.pid_controller(0.1, self._current_velocity[0], 0.1, 0.01, 0.1, self._previous_x_error, self._x_integral, self.control_period)
            # y-axis control
            self._y_acc = self._control.state_feedback_controller(value=self._current_position[1],
                                                                  velocity=self._current_velocity[1],
                                                                  setpoint=self._target_pos[1])
            self._pub_acc_cmd.publish(Vector3(x=float(self._x_acc), y=float(self._y_acc), z=float(0)))

        except Exception as e:
            self._logger.error(f"Control loop error: {e}")
        
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._logger.info(f"[TIMING] Control loop: {elapsed_ms:.2f}ms", throttle_duration_sec=1)

    # Planning callback function
    def planning_callback(self) -> None:
        start_time = time.perf_counter()
        try:
            self._hole = self._planning.find_hole(self._obstacle_map, min_hole_size=40)
            if self._hole is not None:
                self._logger.info(
                    f"Hole found at: {self._hole}", 
                    throttle_duration_sec=1 )
                self._target_pos = self._hole
                if self._perception.mean_x_obstacles(self._perception.compute_obstacles(self._current_laser_scan)) < 0.2:
                    self._logger.info(
                        f"Clearing obstacle map")
                    self._obstacle_map[:] = np.False_
            else:
                self._logger.info(
                    f"No hole found", 
                    throttle_duration_sec=1 )
        except Exception as e:
            self._logger.error(f"Planning loop error: {e}")
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._logger.info(f"[TIMING] Planning loop: {elapsed_ms:.2f}ms", throttle_duration_sec=1)