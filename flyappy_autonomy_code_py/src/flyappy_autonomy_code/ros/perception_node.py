import time
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from flyappy_autonomy_code.core.flyappy_perception import OccupancyGridMapper, PositionEstimator, LaserScanProcessor


class FlyappyPerceptionNode(Node):
    def __init__(self):
        super().__init__("flyappy_perception_node")

        # Parameters and rates
        self.perception_rate = self.declare_parameter("perception_rate", 30.0).value
        self.perception_period = 1.0 / self.perception_rate

        # Perception modules
        self._position_estimator = PositionEstimator()
        self._occupancy_grid_mapper = OccupancyGridMapper()

        # Internal state
        self._current_position = np.array([0.0, 1.0], dtype=np.float64)
        self._current_velocity = np.array([0.0, 0.0], dtype=np.float64)
        self._current_laser_scan: LaserScan | None = None

        # Publisher: estimated position and obstacle map
        self._pub_position = self.create_publisher(Vector3, "/flyappy_estimated_position", 1)
        self._pub_obstacle_map = self.create_publisher(OccupancyGrid, "/flyappy_obstacle_map", 1)  

        self._scan_processor = LaserScanProcessor()
        # Subscribers: velocity and laser scan from Flyappy game
        self._sub_vel = self.create_subscription(Vector3, "/flyappy_vel", self.velocity_callback, 10)
        self._sub_laser_scan = self.create_subscription(LaserScan, "/flyappy_laser_scan", self.laser_scan_callback, 10)
        self._sub_game_ended = self.create_subscription(Bool, "/flyappy_game_ended", self.game_ended_callback, 5)

        # Perception loop timer
        self.perception_timer = self.create_timer(self.perception_period, self.perception_callback)
        
        # TF broadcaster for world -> map transform for RViz
        self._tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(f"Perception node started at {self.perception_rate} Hz.")

    def velocity_callback(self, msg: Vector3) -> None:
        self._current_velocity = np.array([msg.x, msg.y], dtype=np.float64)
        #self.get_logger().info(f"Velocity: [{msg.x},{msg.y},{msg.z}]", throttle_duration_sec=1)

    def laser_scan_callback(self, msg: LaserScan) -> None:
        self._current_laser_scan = msg
        #self.get_logger().info(
        #    f"Laser range: {msg.ranges[0]:.2f}, angle: {msg.angle_min:.2f}",
        #    throttle_duration_sec=1,
        #)

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

            # Broadcast world -> map transform so RViz can keep a stable world frame
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "world"
            transform.child_frame_id = "map"
            transform.transform.translation.x = float(self._occupancy_grid_mapper.map_origin[0])
            transform.transform.translation.y = float(self._occupancy_grid_mapper.map_origin[1])
            transform.transform.translation.z = 0.0
            transform.transform.rotation.w = 1.0
            self._tf_broadcaster.sendTransform(transform)

            # Update map with laser scan if available
            if self._current_laser_scan is not None:
                self._occupancy_grid_mapper.map_add_scan(self._current_position, self._current_laser_scan)

            # Get current obstacle map as OccupancyGrid message
            obstacle_map_msg = self._occupancy_grid_mapper.map_get()
            
            # Set timestamp
            obstacle_map_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Publish obstacle map
            self._pub_obstacle_map.publish(obstacle_map_msg)

            # Publish estimated position
            self._pub_position.publish(Vector3(x=float(self._current_position[0]), y=float(self._current_position[1]), z=0.0))

            #self.get_logger().info(
            #    f"Estimated position: [{self._current_position[0]:.2f},{self._current_position[1]:.2f}]",
            #    throttle_duration_sec=1,
            #)

        except Exception as e:
            self.get_logger().error(f"Perception loop error: {e}")

        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.get_logger().debug(f"[TIMING] Perception loop: {elapsed_ms:.2f}ms", throttle_duration_sec=1)


def main(args=None) -> None:

    rclpy.init(args=args)
    node = FlyappyPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt: Perception node shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()