#!/usr/bin/env python3

import rclpy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool

from flyappy_autonomy_code.flyappy_ros import FlyappyRos


def main() -> None:
    rclpy.init()
    node = rclpy.node.Node('flyappy_autonomy_code_py')
    FlyappyRos(node)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
