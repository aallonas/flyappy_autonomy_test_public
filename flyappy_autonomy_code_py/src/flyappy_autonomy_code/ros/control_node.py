import time
import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool

from flyappy_autonomy_code.core.flyappy_control import ProportionalController, StateFeedbackController


class FlyappyControlNode(Node):
    def __init__(self):
        super().__init__("flyappy_control_node")

        # Parameters and rates
        self.control_rate = self.declare_parameter("control_rate", 30.0).value
        self.control_period = 1.0 / self.control_rate

        # Controllers
        self._proportional_controller = ProportionalController()
        self._state_feedback_controller = StateFeedbackController()

        # Internal state
        self._current_velocity = np.array([0.0, 0.0], dtype=np.float64)
        self._current_position = np.array([0.0, 1.0], dtype=np.float64)
        self._target_pos: tuple[float, float] = (0.0, 0.0)

        # Publisher: acceleration command
        self._pub_acc_cmd = self.create_publisher(Vector3, "/flyappy_acc", 1)

        # Subscribers: velocity, estimated position (from perception), target position (from planning), game events
        self._sub_vel = self.create_subscription(Vector3, "/flyappy_vel", self.velocity_callback, 10)
        self._sub_position = self.create_subscription(Vector3, "/flyappy_estimated_position", self.position_callback, 10)
        self._sub_target = self.create_subscription(Vector3, "/flyappy_target_pos", self.target_pos_callback, 10)
        self._sub_game_ended = self.create_subscription(Bool, "/flyappy_game_ended", self.game_ended_callback, 5)

        # Control loop timer
        self.control_timer = self.create_timer(self.control_period, self.control_callback)

        self.get_logger().info(f"Control node started at {self.control_rate} Hz.")

    def velocity_callback(self, msg: Vector3) -> None:
        self._current_velocity = np.array([msg.x, msg.y], dtype=np.float64)
        #self.get_logger().info(f"Velocity: [{msg.x},{msg.y},{msg.z}]", throttle_duration_sec=1)

    def position_callback(self, msg: Vector3) -> None:
        self._current_position = np.array([msg.x, msg.y], dtype=np.float64)
        #self.get_logger().info(f"Estimated position: [{msg.x},{msg.y}]", throttle_duration_sec=1)

    def target_pos_callback(self, msg: Vector3) -> None:
        self._target_pos = (msg.x, msg.y)
        #self.get_logger().info(f"Target position: [{msg.x},{msg.y}]", throttle_duration_sec=1)

    def game_ended_callback(self, msg: Bool) -> None:
        if msg.data:
            self.get_logger().info("Crash detected.", throttle_duration_sec=1)
        else:
            self.get_logger().info("End of countdown.", throttle_duration_sec=1)

        # Reset controllers and state
        self._proportional_controller.reset()
        self._state_feedback_controller.reset()

        self._current_velocity = np.array([0.0, 0.0], dtype=np.float64)
        self._current_position = np.array([0.0, 1.0], dtype=np.float64)
        self._target_pos = (0.0, 0.0)

    def control_callback(self) -> None:
        start_time = time.perf_counter()
        try:
            # X-axis: keep forward speed at 2.0 using proportional controller
            x_acc = self._proportional_controller.control_step(
                setpoint=1.8,
                velocity=self._current_velocity[0],
                dt=self.control_period,
            )

            # Y-axis: track target y using state feedback controller
            y_acc = self._state_feedback_controller.control_step(
                setpoint=self._target_pos[1],
                position=self._current_position[1],
                velocity=self._current_velocity[1],
            )

            self._pub_acc_cmd.publish(Vector3(x=float(x_acc), y=float(y_acc), z=0.0))
            
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.get_logger().debug(f"[TIMING] Control loop: {elapsed_ms:.2f}ms", throttle_duration_sec=1)


def main(args=None) -> None:
    
    rclpy.init(args=args)
    node = FlyappyControlNode()
    try:  
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt: Control node shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()