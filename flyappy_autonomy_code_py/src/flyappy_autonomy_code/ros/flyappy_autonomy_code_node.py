#!/usr/bin/env python3
"""
This is the main entry point for one of the 2 Flyappy Autonomy Solutions.

It initializes one ROS2 node in which all functionalities are contained 
in the FlyappyRos wrapper elminiating the overhead used to communicate 
between different nodes.

To launch this node, use the command inside the vistual environment: 

    flyappy_autonomy_code_py_node 

"""
import rclpy

from flyappy_autonomy_code.ros.flyappy_ros import FlyappyRos


def main() -> None:
    rclpy.init()
    node = rclpy.node.Node('flyappy_autonomy_code_py')
    FlyappyRos(node)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt: node shutting down..')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
