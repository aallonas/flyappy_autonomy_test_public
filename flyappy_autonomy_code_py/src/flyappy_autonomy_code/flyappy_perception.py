from sensor_msgs.msg import LaserScan
import numpy as np
from numpy import typing as npt
from numpy.typing import NDArray
from typing import Iterator, Tuple

class FlyappyPerception:

    def __init__(self):
        pass
 
    def compute_position(self, current_velocity, previous_position, timestep) -> tuple[float, float]:
        x_pos = previous_position[0] + current_velocity[0] * timestep
        y_pos = previous_position[1] + current_velocity[1] * timestep
        return (x_pos, y_pos)
    
    def compute_obstacles(self, laser_scan: LaserScan, ) -> list[tuple[float, float]]:
        obstacles = []
        angle_min = laser_scan.angle_min
        angle_increment = laser_scan.angle_increment
        for i, distance in enumerate(laser_scan.ranges):
            if  laser_scan.intensities[i] == 1.0:
                angle = angle_min + i * angle_increment
                x_obs = distance * np.cos(angle)
                y_obs = distance * np.sin(angle)
                obstacles.append((x_obs, y_obs))
        return obstacles
    
    # Projection of obstacles in 1D array
    def add_obstacles_to_map(self,
                             obstacle_map: npt.NDArray[np.uint8],
                             obstacles: list[tuple[float, float]],
                             current_position: tuple[float, float]
                             ) -> npt.NDArray[np.bool_]:
        
        for (x_obs, y_obs) in obstacles:
            idy = int(np.round((y_obs + current_position[1])/0.01))
            obstacle_map[idy] = np.True_

        return obstacle_map

    def mean_x_obstacles(self, obstacles: list[tuple[float, float]]) -> float:
        if not obstacles:
            return 0.0
        xs = np.fromiter((x for x, _ in obstacles), dtype=np.float64)
        return float(xs.mean())
    









