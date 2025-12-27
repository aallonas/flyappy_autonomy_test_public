from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
from flyappy_autonomy_code.core.flyappy_control import ProportionalController, StateFeedbackController
from flyappy_autonomy_code.core.flyappy_perception import OccupancyGridMapper, PositionEstimator
from flyappy_autonomy_code.core.flyappy_planning import GapFinder
import numpy as np
from numpy import typing as npt
from typing import Optional
import time

import cv2

class FlyappyRos:
    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()

        self.debug_mode = True

        self._current_position: npt.NDArray[np.float64] = np.array([0.0, 1.0])
        self._previous_position: npt.NDArray[np.float64] = np.array([0.0, 1.0])

        self._obstacle_map: npt.NDArray[np.uint8] = np.zeros((450,), dtype=np.uint8)

        self._hole: Optional[tuple[float, float]] = None  
        self._target_pos: Optional[tuple[float, float]] = (0.0, 0.0)

        # Rate parameters for timers 
        self.control_rate = node.declare_parameter("control_rate", 30.0).value
        self.perception_rate = node.declare_parameter("perception_rate", 30.0).value
        self.planning_rate = node.declare_parameter("planning_rate", 30.0).value
        self.perception_period = 1.0 / self.perception_rate
        self.control_period = 1.0 / self.control_rate
        self.planning_period = 1.0 / self.planning_rate
        
        self._position_estimator = PositionEstimator()
        self._proportional_controller = ProportionalController()
        self._state_feedback_controller = StateFeedbackController()
        self._occupancy_grid_mapper = OccupancyGridMapper()
        self._gap_finder = GapFinder(slice_pixel_size=(20, 190))

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
            f"Flyappy ROS wrapper initialized with:\n"
            f"{self.control_rate}Hz control loop.\n"
            f"{self.perception_rate}Hz perception loop.\n"
            f"{self.planning_rate}Hz planning loop.\n"
            f"debug mode set to {self.debug_mode}.\n"
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
        
        # Reset position estimator and mapping
        self._position_estimator.reset()
        self._occupancy_grid_mapper.reset()

        # Reset controllers
        self._proportional_controller.reset()
        self._state_feedback_controller.reset()

        # Reset current and previous position and velocity
        self._current_position = np.array([0.0, 1.0])
        self._previous_position = np.array([0.0, 1.0])
        self._current_velocity = np.array([0.0, 0.0])   

        # Reset target position
        self._target_pos = (0.0, 0.0)
        self._hole = None

        # Reset sensor data
        self._current_laser_scan = None

    # Perception callback function
    def perception_callback(self) -> None:
        # Start timing for performance monitoring
        start_time = time.perf_counter()

        try:
            # Update position estimation
            self._position_estimator.update_position(current_velocity=(self._current_velocity[0],self._current_velocity[1]),
                                                     timestep=self.perception_period
                                                     )
            # Store current position
            self._current_position = np.array(self._position_estimator.get_position())
            self._logger.info(f"Estimated position: {self._current_position}", throttle_duration_sec=1)
            # roll map
            self._occupancy_grid_mapper.map_roll(self._current_position)

            # Update with laser scan
            self._occupancy_grid_mapper.map_add_scan(self._current_position,
                                                     self._current_laser_scan
                                                     )
            
            # Store current obstacle map
            self._obstacle_map = self._occupancy_grid_mapper.map_get()
            
            # Debug map display
            if self.debug_mode:
                display_map = self._occupancy_grid_mapper.map_get()
                cv2.imshow("Obstacle Map", display_map)
                cv2.waitKey(1)
  
        except Exception as e:
            self._logger.error(f"Perception loop error: {e}")
        
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._logger.info(f"[TIMING] Perception loop: {elapsed_ms:.2f}ms", throttle_duration_sec=1)

    # Control callback function
    def control_callback(self) -> None:
        # Start timing for performance monitoring
        start_time = time.perf_counter()

        try:
            # compute x-axis command
            self._x_acc = self._proportional_controller.control_step(setpoint=2.00,
                                                                    velocity=self._current_velocity[0],
                                                                    dt=self.control_period
                                                                    )
            # compute y-axis command 
            self._y_acc = self._state_feedback_controller.control_step(setpoint=self._target_pos[1],
                                                                      position=self._current_position[1],
                                                                      velocity=self._current_velocity[1]
                                                                      )
            # Publish acceleration command
            self._pub_acc_cmd.publish(Vector3(x=float(self._x_acc), y=float(self._y_acc), z=float(0)))

            self._logger.info(f" target_pos: {self._target_pos}, current_pos: {self._current_position}", throttle_duration_sec=1)
        except Exception as e:
            self._logger.error(f"Control loop error: {e}")
        
        finally:
            # Log timing information
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._logger.info(f"[TIMING] Control loop: {elapsed_ms:.2f}ms", throttle_duration_sec=1)

    # Planning callback function
    def planning_callback(self) -> None:
        start_time = time.perf_counter()
        try:
            self._pixel_position = self._occupancy_grid_mapper.map_position_to_index((self._current_position[0], self._current_position[1]))
            self._hole = self._gap_finder.find_free_rows(self._obstacle_map,
                                                         self._pixel_position,
                                                         )
            
            self._target_pos = self._occupancy_grid_mapper.map_index_to_position((0, int(self._hole[1])))

            self._logger.info(f"Hole found at: {self._hole} converted to position: {self._target_pos}", throttle_duration_sec=1)

        except Exception as e:
            self._logger.error(f"Planning loop error: {e}")

        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._logger.info(f"[TIMING] Planning loop: {elapsed_ms:.2f}ms", throttle_duration_sec=1)