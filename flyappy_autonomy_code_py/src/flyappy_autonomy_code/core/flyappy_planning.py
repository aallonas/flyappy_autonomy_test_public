import numpy as np
from numpy import typing as npt




class GapFinder:

    """ Class for the gap finding algorithm based on the occupancy grid map.
    The algorithm searches for free rows in a specified vertical slice of the map
    where the bird can pass through without colliding with obstacles.
    """


    def __init__(
            self,
            bird_pixel_size: float = 32.0,
            slice_pixel_size: tuple[int, int] = (0, 160)
            ) -> None:
        """ Initialize the gap finder.
        
        Args:
            bird_pixel_size (float): The size of the bird in pixels.
            slice_pixel_size (tuple[int, int]): The slice of the map to consider (start_row, end_row).
        """
        self.bird_pixel_size = bird_pixel_size
        self.slice_pixel_size = slice_pixel_size
    
    
    def find_free_rows(
            self,
            obstacle_map: npt.NDArray[np.int8],
            map_position: tuple[int, int],
            ) -> tuple[int, int]:
        """ Find free rows in the obstacle map for the bird to pass through.

        1. Extract the relevant slice of the obstacle map.
        2. Inflate obstacles in y direction based on the bird's size.
        3. Identify free rows in the inflated map.
        4. Select the best gap based on proximity to the current bird position.
        
        Args:
            obstacle_map (npt.NDArray[np.int8]): The occupancy grid map as a 2D array. 
                Uses ROS standard: -1=UNKNOWN, 0=FREE, 100=OCCUPIED.
            map_position (tuple[int, int]): The current pixel position of the bird (x, y).

        Returns:
            tuple[int, int]: The (x, y) position of the center of the best gap found in map coordinates.
        
        """
        # Step 1: Extract the relevant slice of the obstacle map
        x_pos, y_pos = map_position
        slice_map = obstacle_map[self.slice_pixel_size[0]:self.slice_pixel_size[1], :]
        
        # Step 2: Inflate obstacles based on bird size
        bird_radius = int(np.ceil(self.bird_pixel_size / 2))
        inflated = slice_map.copy()
        
        # Inflate by rolling occupied cells (100)
        for dy in range(-bird_radius, bird_radius + 1):
            rolled = np.roll(slice_map == 100, shift=dy, axis=1)
            inflated[rolled] = 100
        
        # Step 3: Identify free rows in the inflated map
        # A row (y position) is considered free if all columns in that row are free (0) or unknown (-1)
        free_rows_mask = np.all(inflated < 1, axis=0)
        free_row_indices = np.where(free_rows_mask)[0]
        
        # Handle case with no free rows
        if free_row_indices.size == 0:
            # Add exploration behavior here if needed
            return 0, y_pos
        
        # Step 4: Find groups of free rows (gaps)
        gaps = []
        start = free_row_indices[0]
        
        # Iterate through free row indices to find gaps
        for i in range(1, len(free_row_indices)):
            # Check if indexes are consecutive
            if free_row_indices[i] != free_row_indices[i - 1] + 1:
                gaps.append((start, free_row_indices[i - 1]))
                start = free_row_indices[i]
        
        # Add the last gap
        gaps.append((start, free_row_indices[-1]))
        
        # Select gap closest to current y position
        best_gap = min(
            gaps,
            key=lambda g: abs((g[0] + g[1]) / 2 - y_pos)
        )
        
        # Return center of best gap
        y_center = int((best_gap[0] + best_gap[1]) / 2)
        
        return 0, y_center
