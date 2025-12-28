import time
from typing import Optional

import numpy as np
from numpy import typing as npt
from geometry_msgs.msg import TransformStamped, Vector3
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster

from flyappy_autonomy_code.core.flyappy_control import ProportionalController, StateFeedbackController
from flyappy_autonomy_code.core.flyappy_perception import OccupancyGridMapper, PositionEstimator
from flyappy_autonomy_code.core.flyappy_planning import GapFinder


class FlyappyRos:
    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()

        # Parameters and derived periods
        self.control_rate = node.declare_parameter("control_rate", 60.0).value
        self.perception_rate = node.declare_parameter("perception_rate", 30.0).value
        self.planning_rate = node.declare_parameter("planning_rate", 30.0).value
        self.control_period = 1.0 / self.control_rate
        self.perception_period = 1.0 / self.perception_rate
        self.planning_period = 1.0 / self.planning_rate

        # Core modules
        self._position_estimator = PositionEstimator()
        self._occupancy_grid_mapper = OccupancyGridMapper()
        self._gap_finder = GapFinder(slice_pixel_size=(20, 190))
        self._proportional_controller = ProportionalController()
        self._state_feedback_controller = StateFeedbackController()

        # Internal state shared across perception, planning, and control
        self._current_position: npt.NDArray[np.float64] = np.array([0.0, 1.0], dtype=np.float64)
        self._current_velocity: npt.NDArray[np.float64] = np.array([0.0, 0.0], dtype=np.float64)
        self._current_laser_scan: Optional[LaserScan] = None
        self._obstacle_map: npt.NDArray[np.int8] = self._occupancy_grid_mapper.obstacle_map
        self._target_pos: tuple[float, float] = (0.0, 0.0)
        self._hole: Optional[tuple[float, float]] = None
        self._perception_max_ms: float = 0.0

        # TF broadcaster for world -> map so RViz can visualize the rolling map
        self._tf_broadcaster = TransformBroadcaster(node)

        # IO with the Flyappy game
        self._pub_acc_cmd = node.create_publisher(Vector3, "/flyappy_acc", 1)
        self._pub_obstacle_map = node.create_publisher(OccupancyGrid, "/flyappy_obstacle_map", 1)

        self._sub_vel = node.create_subscription(Vector3, "/flyappy_vel", self.velocity_callback, 10)
        self._sub_laser_scan = node.create_subscription(LaserScan, "/flyappy_laser_scan", self.laser_scan_callback, 10)
        self._sub_game_ended = node.create_subscription(Bool, "/flyappy_game_ended", self.game_ended_callback, 5)

        # Timers for control, perception, and planning loops (shared in-process, no inter-node topics)
        self.control_timer = node.create_timer(self.control_period, self.control_callback)
        self.perception_timer = node.create_timer(self.perception_period, self.perception_callback)
        self.planning_timer = node.create_timer(self.planning_period, self.planning_callback)

        self._logger.info(
            f"Flyappy ROS wrapper initialized at {self.control_rate}Hz control, "
            f"{self.perception_rate}Hz perception, {self.planning_rate}Hz planning."
        )

    def velocity_callback(self, msg: Vector3) -> None:
        self._current_velocity = np.array([msg.x, msg.y], dtype=np.float64)

    def laser_scan_callback(self, msg: LaserScan) -> None:
        self._current_laser_scan = msg

    def game_ended_callback(self, msg: Bool) -> None:
        if msg.data:
            self._logger.info(f"Crash detected, received: {msg.data}")
        else:
            self._logger.info(f"End of countdown, received: {msg.data}")
        
        # Reset position estimator and mapping
        self._position_estimator.reset()
        self._occupancy_grid_mapper.reset()

        # Reset controllers
        self._proportional_controller.reset()
        self._state_feedback_controller.reset()

        # Reset state
        self._current_position = np.array([0.0, 1.0], dtype=np.float64)
        self._current_velocity = np.array([0.0, 0.0], dtype=np.float64)
        self._current_laser_scan = None
        self._target_pos = (0.0, 0.0)
        self._hole = None

        # Keep obstacle map reference consistent
        self._obstacle_map = self._occupancy_grid_mapper.obstacle_map

    # Perception callback function
    def perception_callback(self) -> None:
        start_time = time.perf_counter()

        try:
            # Update position estimation from velocity
            self._position_estimator.update_position(
                current_velocity=(self._current_velocity[0], self._current_velocity[1]),
                timestep=self.perception_period,
            )
            self._current_position = np.array(self._position_estimator.get_position(), dtype=np.float64)

            # Roll map forward and integrate latest scan
            self._occupancy_grid_mapper.map_roll(self._current_position)
            if self._current_laser_scan is not None:
                self._occupancy_grid_mapper.map_add_scan(self._current_position, self._current_laser_scan)

            # Cache latest obstacle map for planning
            self._obstacle_map = self._occupancy_grid_mapper.obstacle_map

            # Broadcast TF for RViz
            transform = TransformStamped()
            transform.header.stamp = self._node.get_clock().now().to_msg()
            transform.header.frame_id = "world"
            transform.child_frame_id = "map"
            transform.transform.translation.x = float(self._occupancy_grid_mapper.map_origin[0])
            transform.transform.translation.y = float(self._occupancy_grid_mapper.map_origin[1])
            transform.transform.translation.z = 0.0
            transform.transform.rotation.w = 1.0
            self._tf_broadcaster.sendTransform(transform)

            # Publish occupancy grid for visualization/debugging
            obstacle_map_msg = self._occupancy_grid_mapper.map_get()
            obstacle_map_msg.header.stamp = self._node.get_clock().now().to_msg()
            self._pub_obstacle_map.publish(obstacle_map_msg)

        except Exception as e:
            self._logger.error(f"Perception loop error: {e}")
        
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._perception_max_ms = max(self._perception_max_ms, elapsed_ms)
            self._logger.debug(
                f"[TIMING] Perception loop: {elapsed_ms:.2f}ms (max: {self._perception_max_ms:.2f}ms)",
                throttle_duration_sec=1,
            )

    # Control callback function
    def control_callback(self) -> None:
        start_time = time.perf_counter()

        try:
            x_acc = self._proportional_controller.control_step(
                setpoint=1.8,
                velocity=self._current_velocity[0],
                dt=self.control_period,
            )
            y_acc = self._state_feedback_controller.control_step(
                setpoint=self._target_pos[1],
                position=self._current_position[1],
                velocity=self._current_velocity[1],
            )

            self._pub_acc_cmd.publish(Vector3(x=float(x_acc), y=float(y_acc), z=0.0))
        except Exception as e:
            self._logger.error(f"Control loop error: {e}")
        
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._logger.debug(f"[TIMING] Control loop: {elapsed_ms:.2f}ms", throttle_duration_sec=1)

    # Planning callback function
    def planning_callback(self) -> None:
        start_time = time.perf_counter()
        try:
            pixel_position = self._occupancy_grid_mapper.map_position_to_index(
                (self._current_position[0], self._current_position[1])
            )
            self._hole = self._gap_finder.find_free_rows(self._obstacle_map, pixel_position)

            self._target_pos = self._occupancy_grid_mapper.map_index_to_position((0, int(self._hole[1])))

        except Exception as e:
            self._logger.error(f"Planning loop error: {e}")

        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._logger.debug(f"[TIMING] Planning loop: {elapsed_ms:.2f}ms", throttle_duration_sec=1)