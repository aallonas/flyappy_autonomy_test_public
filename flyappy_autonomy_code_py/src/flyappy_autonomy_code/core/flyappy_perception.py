from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class LaserScanProcessor:
    """Processes LaserScan messages into relative coordinates.

    Caches angle, cos, and sin arrays to avoid recomputation when scan size and
    angular parameters stay the same between messages.
    """

    def __init__(self) -> None:
        self._cached_len: int | None = None
        self._cached_angle_min: float | None = None
        self._cached_angle_inc: float | None = None
        self._angles: NDArray[np.float64] | None = None
        self._cos: NDArray[np.float64] | None = None
        self._sin: NDArray[np.float64] | None = None

    def _update_cache(self, count: int, angle_min: float, angle_inc: float) -> None:
        self._cached_len = count
        self._cached_angle_min = angle_min
        self._cached_angle_inc = angle_inc
        self._angles = angle_min + np.arange(count, dtype=np.float64) * angle_inc
        self._cos = np.cos(self._angles)
        self._sin = np.sin(self._angles)

    def process_scan(self,
                     laser_scan: LaserScan
                     ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """ Process laser scan data into numpy arrays of relative cartesian coordinates.

        Args:
            laser_scan (LaserScan): Input laser scan data.
        
        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: 
                - First array: Nx2 array of hit coordinates (obstacles) [[x, y], ...]
                - Second array: Mx2 array of miss coordinates (free space) [[x, y], ...]
        """
        ranges = np.asarray(laser_scan.ranges, dtype=np.float64)
        count = len(ranges)

        # Refresh cached trigonometry if scan geometry changed
        if (
            self._cached_len != count
            or self._cached_angle_min != laser_scan.angle_min
            or self._cached_angle_inc != laser_scan.angle_increment
        ):
            self._update_cache(count, laser_scan.angle_min, laser_scan.angle_increment)

        rel_x = ranges * self._cos
        rel_y = ranges * self._sin
        intensities = np.asarray(laser_scan.intensities, dtype=np.float64)

        # Separate hits (intensity > 0) from misses (intensity == 0)
        hit_mask = intensities > 0.0
        miss_mask = ~hit_mask

        hits = np.column_stack((rel_x[hit_mask], rel_y[hit_mask]))
        misses = np.column_stack((rel_x[miss_mask], rel_y[miss_mask]))

        return hits, misses


class OccupancyGridMapper:
    """ Class for managing an occupancy grid map based on laser scan data. 
    The map is represented as a 2D numpy array with occupancy values:
    - OCCUPIED (100): Cell is occupied by an obstacle.
    - FREE (0): Cell is free space.
    - UNKNOWN (-1): Cell has not been observed yet.

    Functionality:
    -The map can be rolled forward as the agent moves with: map_roll()
    -New laser scan data can be integrated into the map with: map_add_scan()
    -The map can be reset to unknown values with: map_reset()
    -Conversion between world positions and map indices is provided with:
        map_position_to_index() and map_index_to_position()
    -The current map can be obtained as an OccupancyGrid message with: map_get()
    """

    # Occupancy values (ROS standard)
    OCCUPIED = 100
    FREE = 0
    UNKNOWN = -1

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
            map_origin (Tuple[float, float]): Origin of the map in world coordinates (x, y) [m].
            agent_size (int): Size of the agent in map cells (assumed square).
        """
        self.map_resolution = map_resolution
        self.map_size = map_size
        self.map_origin = map_origin
        self.agent_size = agent_size  

        # Reusable scan processor to avoid per-call allocations
        self._scan_processor = LaserScanProcessor()

        # Initialize map with unknown values (int8 for ROS compatibility)
        self.obstacle_map = np.full(self.map_size, fill_value=self.UNKNOWN, dtype=np.int8)


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
                     position: tuple[float, float],
                     laser_scan: LaserScan
                     ) -> None:
        """ Add a laser scan to the occupancy grid map.

        Args:
            position (tuple[float, float]): Current position (x, y).
            laser_scan (LaserScan): Laser scan data.
        """
        # Only update if laser scan is available
        if laser_scan is None:
            return

        # Process laser scan to get hit and miss coordinates
        hits, misses = self._scan_processor.process_scan(laser_scan)

        # Sensor position in grid coordinates (relative to map origin)
        sensor_x = position[0] - self.map_origin[0] + self.map_resolution * self.agent_size
        sensor_y = position[1] - self.map_origin[1]

        # Sensor position in map indices
        x0 = int(sensor_x / self.map_resolution)
        y0 = int(sensor_y / self.map_resolution)

        H, W = self.map_size

        # Process misses (free space)
        for miss in misses:
            # Endpoint relative to map origin
            end_x = sensor_x + miss[0]
            end_y = sensor_y + miss[1]

            # Endpoint in map indices
            x1 = int(end_x / self.map_resolution)
            y1 = int(end_y / self.map_resolution)

            # Get cells traversed by the laser beam
            cells = self.raycast_dda((x0, y0), (x1, y1))

            # Check if cells are within map bounds
            valid = (
                (cells[:, 0] >= 0) & (cells[:, 0] < H) &
                (cells[:, 1] >= 0) & (cells[:, 1] < W)
            )
            cells = cells[valid]

            # Mark ONLY unknown cells as free
            unknown = self.obstacle_map[cells[:, 0], cells[:, 1]] == self.UNKNOWN
            self.obstacle_map[cells[unknown, 0], cells[unknown, 1]] = self.FREE

        # Process hits (obstacles)
        for hit in hits:
            # Endpoint relative to map origin
            end_x = sensor_x + hit[0]
            end_y = sensor_y + hit[1]

            # Endpoint in map indices
            x1 = int(end_x / self.map_resolution)
            y1 = int(end_y / self.map_resolution)

            # Get cells traversed by the laser beam (mark as free)
            cells = self.raycast_dda((x0, y0), (x1, y1))

            # Check if cells are within map bounds
            valid = (
                (cells[:, 0] >= 0) & (cells[:, 0] < H) &
                (cells[:, 1] >= 0) & (cells[:, 1] < W)
            )
            cells = cells[valid]

            # Mark ONLY unknown cells as free (except the endpoint)
            if len(cells) > 1:  # Ensure there's more than just the endpoint
                unknown = self.obstacle_map[cells[:-1, 0], cells[:-1, 1]] == self.UNKNOWN
                self.obstacle_map[cells[:-1][unknown, 0], cells[:-1][unknown, 1]] = self.FREE

            # Mark the endpoint as occupied if within bounds
            if 0 <= x1 < H and 0 <= y1 < W:
                self.obstacle_map[x1, y1] = self.OCCUPIED


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


    def reset(
            self
            ) -> None:
        """ Reset the occupancy grid map to unknown values and origin to (0,0). """
        self.obstacle_map.fill(self.UNKNOWN)
        self.map_origin = (0.0, 0.0)



    def map_get(self) -> OccupancyGrid:
        """ Get the current occupancy grid map as a ROS OccupancyGrid message.
        
        Returns:
            OccupancyGrid: Current occupancy grid map.
        """
        grid_msg = OccupancyGrid()
        
        # Header
        grid_msg.header.frame_id = "map"
        # Note: timestamp should be set by the node publishing this
        
        # Map metadata
        grid_msg.info.resolution = float(self.map_resolution)
        grid_msg.info.width = self.map_size[1]  # Width in cells
        grid_msg.info.height = self.map_size[0]  # Height in cells
        
        # Origin in the map frame: keep at (0,0) because the map frame itself moves via TF
        grid_msg.info.origin.position.x = 0.0
        grid_msg.info.origin.position.y = 0.0
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0  # No rotation
        
        # Map data (row-major order, flattened)
        # ROS uses [-1, 0, 100] format which matches our internal representation
        grid_msg.data = self.obstacle_map.flatten().tolist()
        
        return grid_msg


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
            self,
            initial_position: Tuple[float, float] = (0.0, 1.5)
            )-> None:
        """" Initialize the position estimator.
        
        Args:
            initial_position (Tuple[float, float]): Initial position (x, y).
        """
        self.current_position: Tuple[float, float] = initial_position


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


    def reset(
            self
            ) -> None:
        """ Reset the position estimator to the origin. """
        self.current_position = (0.0, 1.5)