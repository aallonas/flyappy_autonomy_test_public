import time

import numpy as np
import cv2
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool

from flyappy_autonomy_code.core.flyappy_perception import OccupancyGridMapper, PositionEstimator


class FlyappyPerceptionNode(Node):
    def __init__(self):
        super().__init__("flyappy_perception_node")

        # Parameters and rates
        self.perception_rate = self.declare_parameter("perception_rate", 30.0).value
        self.perception_period = 1.0 / self.perception_rate
        self.debug_mode = self.declare_parameter("debug_mode", True).value

        # Perception modules
        self._position_estimator = PositionEstimator()
        self._occupancy_grid_mapper = OccupancyGridMapper()

        # Internal state
        self._current_position = np.array([0.0, 1.0], dtype=np.float64)
        self._current_velocity = np.array([0.0, 0.0], dtype=np.float64)
        self._current_laser_scan: LaserScan | None = None
        self._obstacle_map = np.zeros((450,), dtype=np.uint8)

        # Publisher: estimated position and obstacle map
        self._pub_position = self.create_publisher(Vector3, "/flyappy_estimated_position", 1)
        self._pub_obstacle_map = self.create_publisher(UInt8MultiArray, "/flyappy_obstacle_map", 1)  

        # Subscribers: velocity and laser scan from Flyappy game
        self._sub_vel = self.create_subscription(Vector3, "/flyappy_vel", self.velocity_callback, 10)
        self._sub_laser_scan = self.create_subscription(LaserScan, "/flyappy_laser_scan", self.laser_scan_callback, 10)
        self._sub_game_ended = self.create_subscription(Bool, "/flyappy_game_ended", self.game_ended_callback, 5)

        # Perception loop timer
        self.perception_timer = self.create_timer(self.perception_period, self.perception_callback)

        self.get_logger().info(f"Perception node started at {self.perception_rate} Hz. Debug mode: {self.debug_mode}")

    def velocity_callback(self, msg: Vector3) -> None:
        self._current_velocity = np.array([msg.x, msg.y], dtype=np.float64)
        self.get_logger().info(f"Velocity: [{msg.x},{msg.y},{msg.z}]", throttle_duration_sec=1)

    def laser_scan_callback(self, msg: LaserScan) -> None:
        self._current_laser_scan = msg
        self.get_logger().info(
            f"Laser range: {msg.ranges[0]:.2f}, angle: {msg.angle_min:.2f}",
            throttle_duration_sec=1,
        )

    def game_ended_callback(self, msg: Bool) -> None:
        if msg.data:
            self.get_logger().info("Crash detected.", throttle_duration_sec=1)
        else:
            self.get_logger().info("End of countdown.", throttle_duration_sec=1)

        # Reset perception modules
        self._position_estimator.reset()
        self._occupancy_grid_mapper.reset()

        # Reset state
        self._current_position = np.array([0.0, 1.0], dtype=np.float64)
        self._current_velocity = np.array([0.0, 0.0], dtype=np.float64)
        self._current_laser_scan = None
        self._obstacle_map = np.zeros((450,), dtype=np.uint8)

    def perception_callback(self) -> None:
        start_time = time.perf_counter()

        try:
            # Update position estimation based on velocity
            self._position_estimator.update_position(
                current_velocity=(self._current_velocity[0], self._current_velocity[1]),
                timestep=self.perception_period,
            )
            self._current_position = np.array(self._position_estimator.get_position())

            # Roll occupancy map with current position
            self._occupancy_grid_mapper.map_roll(self._current_position)

            # Update map with laser scan if available
            if self._current_laser_scan is not None:
                self._occupancy_grid_mapper.map_add_scan(self._current_position, self._current_laser_scan)

            # Get current obstacle map
            self._obstacle_map = self._occupancy_grid_mapper.map_get()
            map_msg = UInt8MultiArray()
            map_msg.data = self._obstacle_map.flatten().tolist()
            self._pub_obstacle_map.publish(map_msg)

            # Publish estimated position
            self._pub_position.publish(Vector3(x=float(self._current_position[0]), y=float(self._current_position[1]), z=0.0))

            # Debug map display
            if self.debug_mode:
                display_map = self._occupancy_grid_mapper.map_get()
                cv2.imshow("Obstacle Map", display_map)
                cv2.waitKey(1)

            self.get_logger().info(
                f"Estimated position: [{self._current_position[0]:.2f},{self._current_position[1]:.2f}]",
                throttle_duration_sec=1,
            )

        except Exception as e:
            self.get_logger().error(f"Perception loop error: {e}")

        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.get_logger().info(f"[TIMING] Perception loop: {elapsed_ms:.2f}ms", throttle_duration_sec=1)


def main(args=None) -> None:
    import rclpy

    rclpy.init(args=args)
    node = FlyappyPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user (KeyboardInterrupt).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()