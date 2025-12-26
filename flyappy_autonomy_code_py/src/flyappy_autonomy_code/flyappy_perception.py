from sensor_msgs.msg import LaserScan
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class OccupancyGridMapper:

    """ Class for managing an occupancy grid map based on laser scan data. 
    The map is represented as a 2D numpy array with occupancy values:
    - OCCUPIED (255): Cell is occupied by an obstacle.
    - FREE (0): Cell is free space.
    - UNKNOWN (127): Cell has not been observed yet.

    Functionality:
    -The map can be rolled forward as the agent moves with: map_roll()
    -New laser scan data can be integrated into the map with: map_add_scan()
    -The map can be reset to unknown values with: map_reset()
    -Conversion between world positions and map indices is provided with:
        map_position_to_index() and map_index_to_position()
    -The current map can be obtained with: map_get()
    """

    # Ocupancy values
    OCCUPIED = 255
    FREE = 0
    UNKNOWN = 127

    def __init__(
            self,
            map_resolution: float = 0.01,
            map_size: Tuple[int, int] = (410, 410),
            map_origin: Tuple[float, float] = (0.0, 0.0),
            agent_size: int = 32
            ) -> None:
        """ Initialize the occupancy grid mapper.
        
        Args:
            map_resolution (float): Resolution of the map in meters per cell.
            map_size (Tuple[int, int]): Size of the map in number of cells (height, width).
            origin (Tuple[float, float]): Origin of the map in world coordinates (x, y) [m].
            agent_size (int): Size of the agent in map cells (assumed square).
        """
        self.map_resolution = map_resolution
        self.map_size = map_size
        self.map_origin = map_origin
        self.agent_size = agent_size  

        # Initialize map with unknown values
        self.obstacle_map = np.full(self.map_size, fill_value=self.UNKNOWN, dtype=np.uint8)

    def map_roll(self,
                 position : tuple[float, float],
                 ) -> None:
        """ Check if the map can be rolled based on the current position and the maps origin.
        If so, the map is rolled and the additional area is marked as unknown.

        Args:
            position (tuple[float, float]): Current position (x, y).
                
        """
        # Compute shift in meters (float)
        shift = position[0] - self.map_origin[0]
        # Convert shift in cells (int)
        cell_shift = int(np.floor(shift / self.map_resolution))

        # Check if we need to roll the map
        if cell_shift >=1:
            # Move map origin forward by the rolled distance
            self.map_origin = (
                self.map_origin[0] + cell_shift * self.map_resolution,
                self.map_origin[1]
            )
            
            # Roll map
            self.obstacle_map = np.roll(
                self.obstacle_map, shift=-cell_shift, axis=0
            )

            # Mark unscanned newly revealed area as UNKNOWN
            self.obstacle_map[self.map_size[0]-cell_shift:, :] = self.UNKNOWN


    def map_add_scan(self,
                     position : tuple[float, float],
                     laser_scan : LaserScan
                     ) -> None:
        """ Add a laser scan to the occupancy grid map.
        
        Args:
            position (tuple[float, float]): Current position (x, y).
            laser_scan (LaserScan): Laser scan data.
        
        """
        # Only update if laser scan is available
        if laser_scan is not None:

            # Sensor position in grid coordinates
            rel_x = position[0] - self.map_origin[0] + self.map_resolution * self.agent_size
            rel_y = position[1] - self.map_origin[1]

            # Sensor position in map indices
            x0 = int(rel_x / self.map_resolution)
            y0 = int(rel_y / self.map_resolution)

            H, W = self.map_size

            for i, distance in enumerate(laser_scan.ranges):
            
                angle = laser_scan.angle_min + i * laser_scan.angle_increment

                # Endpoint relative to map origin
                end_x = rel_x + distance * np.cos(angle)
                end_y = rel_y + distance * np.sin(angle)

                # Endpoint in map indices
                x1 = int(end_x / self.map_resolution)
                y1 = int(end_y / self.map_resolution)

                # Get cells that are traversed by the laser beam
                cells = self.raycast_dda((x0, y0), (x1, y1))
                # Cells = self.bresenham((x0, y0), (x1, y1))

                # Check if cells are within map bounds
                valid = (
                        (cells[:, 0] >= 0) & (cells[:, 0] < H) &
                        (cells[:, 1] >= 0) & (cells[:, 1] < W)
                )
                cells = cells[valid]

                # Check if any cells to be marked are unknown
                unknown = self.obstacle_map[cells[:, 0], cells[:, 1]] == 127
                # Mark ONLY unknown cells as free
                self.obstacle_map[cells[unknown, 0], cells[unknown, 1]] = 0

                # If Hit an obstacle, mark the endpoint as occupied
                if laser_scan.intensities[i] > 0.0:
                    # Check if endpoint is within map bounds
                    if 0 <= x1 < H and 0 <= y1 < W:
                        self.obstacle_map[x1, y1] = 255

    def map_position_to_index(self,
                       position: tuple[float, float]
                       ) -> tuple[int, int]:
        """ Convert world position to map index. 
        
        Args:
            position (tuple[float, float]): World position (x, y).

        Returns:
            tuple[int, int]: Map index (i, j).
        """
        return (int(position[0] / self.map_resolution), int(position[1] / self.map_resolution))
    
    def map_index_to_position(self,
                       index: tuple[int, int]
                       ) -> tuple[float, float]:
        """ Convert map index to world position.
    
        Args:
            index (tuple[int, int]): Map index (i, j).
            
        Returns:
            tuple[float, float]: World position (x, y).
        """
        return (index[0] * self.map_resolution, index[1] * self.map_resolution)

    def map_reset(
            self
            ) -> None:
        """ Reset the occupancy grid map to unknown values. """
        self.obstacle_map.fill(127)

    def map_get(
            self
            ) -> NDArray[np.uint8]:
        """ Get the current occupancy grid map. """
        return self.obstacle_map.copy()

# === Line drawing algorithms ===

    def bresenham(self,
                  start : Tuple[int, int],
                  finish: Tuple[int, int]
                  ) -> NDArray[np.int_]:
        # step 1: get line start and end-points 
        x0, y0 = start
        x1, y1 = finish
        # step 2: calculate slope
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        # step 3: determine step size and direction 
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        # step 4: iterate over line
        points = []
        while x0 != x1 or y0 != y1:
            points.append([x0, y0])
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        points.append([x0, y0])
        return np.array(points, dtype=np.int_)
    
    def raycast_dda(self,
                    start: Tuple[int, int],
                    finish: Tuple[int, int]
                    ) -> NDArray[np.int_]:
        x0, y0 = start
        x1, y1 = finish

        dx = x1 - x0
        dy = y1 - y0

        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return np.array([[x0, y0]], dtype=np.int_)

        t = np.linspace(0.0, 1.0, steps + 1)

        xs = np.round(x0 + t * dx).astype(np.int_)
        ys = np.round(y0 + t * dy).astype(np.int_)

        return np.column_stack((xs, ys))
    
# === End of line drawing algorithms ===



class PositionEstimator:

    """ Class for estimating the position of the agent based on velocity inputs. 

    -The position can be updated with: update_position()
    -The current position can be retrieved with: get_position()
    -The position can be set directly with: set_position()
    -The position can be reset to the origin with: reset_position()
    """

    def __init__(
            self
            )-> None:
        self.current_position: Tuple[float, float] = (0.0, 1.5)

    def update_position(
            self,
            current_velocity: Tuple[float, float],
            timestep: float
            ) -> None:
        """ Update position based on velocity at each timestep.

        Args:
            current_velocity (Tuple[float, float]): Current velocity (vx, vy).
            timestep (float): Time step duration.

        Returns:
            Tuple[float, float]: Updated position (x, y).
        """
        if timestep <= 0.0:
            raise ValueError("timestep must be positive")
        
        
        x_pos = self.current_position[0] + current_velocity[0] * timestep
        y_pos = self.current_position[1] + current_velocity[1] * timestep
        self.current_position = (x_pos, y_pos)

    def get_position(
            self
            ) -> Tuple[float, float]:
        """ Get the current position.

        Returns:
            Tuple[float, float]: Current position (x, y).
        """
        return self.current_position

    def set_position(
            self,
            position: Tuple[float, float]
            ) -> None:
        """ Set the current position.

        Args:
            position (Tuple[float, float]): New position (x, y).
        """
        self.current_position = position

    def reset_position(
            self
            ) -> None:
        """ Reset the position estimator to the origin. """
        self.current_position = (0.0, 0.0)