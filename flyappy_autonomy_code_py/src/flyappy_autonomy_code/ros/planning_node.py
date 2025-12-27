import time

import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
from std_msgs.msg import UInt8MultiArray


from flyappy_autonomy_code.core.flyappy_planning import GapFinder
from flyappy_autonomy_code.core.flyappy_perception import OccupancyGridMapper


class FlyappyPlanningNode(Node):
    def __init__(self):
        super().__init__("flyappy_planning_node")

        # Parameters and rates
        self.planning_rate = self.declare_parameter("planning_rate", 30.0).value
        self.planning_period = 1.0 / self.planning_rate

        # Planning modules
        self._gap_finder = GapFinder(slice_pixel_size=(20, 190))
        self._occupancy_grid_mapper = OccupancyGridMapper()

        # Internal state
        self._current_position = np.array([0.0, 1.0], dtype=np.float64)
        self._obstacle_map = np.zeros((410, 410), dtype=np.uint8)
        self._hole = None
        self._target_pos = (0.0, 0.0)

        # Publisher: target position for control node
        self._pub_target_pos = self.create_publisher(Vector3, "/flyappy_target_pos", 1)

        # Subscribers: estimated position and obstacle map from perception node
        self._sub_position = self.create_subscription(
            Vector3, "/flyappy_estimated_position", self.position_callback, 10
        )
        self._sub_obstacle_map = self.create_subscription(
            UInt8MultiArray, "/flyappy_obstacle_map", self.obstacle_map_callback, 10
        )
        self._sub_game_ended = self.create_subscription(Bool, "/flyappy_game_ended", self.game_ended_callback, 5)

        # Planning loop timer
        self.planning_timer = self.create_timer(self.planning_period, self.planning_callback)

        self.get_logger().info(f"Planning node started at {self.planning_rate} Hz.")

    def position_callback(self, msg: Vector3) -> None:
        self._current_position = np.array([msg.x, msg.y], dtype=np.float64)
        self.get_logger().info(
            f"Estimated position: [{msg.x:.2f},{msg.y:.2f}]", throttle_duration_sec=1
        )

    def obstacle_map_callback(self, msg: UInt8MultiArray) -> None:
        self._obstacle_map = np.array(msg.data, dtype=np.uint8).reshape((410, 410))
        self.get_logger().info(
            f"Obstacle map received with shape {self._obstacle_map.shape}", throttle_duration_sec=1
        )

    def game_ended_callback(self, msg: Bool) -> None:
        if msg.data:
            self.get_logger().info("Crash detected.", throttle_duration_sec=1)
        else:
            self.get_logger().info("End of countdown.", throttle_duration_sec=1)

        # Reset planning modules
        self._gap_finder = GapFinder(slice_pixel_size=(20, 190))
        self._occupancy_grid_mapper = OccupancyGridMapper()

        # Reset state
        self._current_position = np.array([0.0, 1.0], dtype=np.float64)
        self._obstacle_map = np.zeros((410, 410), dtype=np.uint8)
        self._hole = None
        self._target_pos = (0.0, 0.0)

    def planning_callback(self) -> None:
        start_time = time.perf_counter()

        try:
            # Convert current position to pixel coordinates
            self._pixel_position = self._occupancy_grid_mapper.map_position_to_index(
                (self._current_position[0], self._current_position[1])
            )

            # Find the free gap in the obstacle map
            self._hole = self._gap_finder.find_free_rows(
                self._obstacle_map, self._pixel_position
            )

            # Convert pixel coordinates back to world position
            self._target_pos = self._occupancy_grid_mapper.map_index_to_position(
                (0, int(self._hole[1]))
            )

            # Publish target position
            self._pub_target_pos.publish(
                Vector3(x=float(self._target_pos[0]), y=float(self._target_pos[1]), z=0.0)
            )

            self.get_logger().info(
                f"Hole found at: {self._hole} converted to position: {self._target_pos}",
                throttle_duration_sec=1,
            )

        except Exception as e:
            self.get_logger().error(f"Planning loop error: {e}")

        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.get_logger().info(
                f"[TIMING] Planning loop: {elapsed_ms:.2f}ms", throttle_duration_sec=1
            )


def main(args=None) -> None:
    import rclpy

    rclpy.init(args=args)
    node = FlyappyPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user (KeyboardInterrupt).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()