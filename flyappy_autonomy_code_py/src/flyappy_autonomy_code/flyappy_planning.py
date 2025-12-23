import numpy as np
from numpy import typing as npt

class FlyappyPlanning:

    def __init__(self):
        pass

    def find_hole(self,
                  obstacle_map: npt.NDArray[np.bool_],
                  min_hole_size: int = 50
                  ) -> tuple[int, int] | None:
        #Find the center of the largest continuous gap in the obstacle map
        if np.count_nonzero(obstacle_map) > len(obstacle_map)/3:
            # Find continuous regions of False (no obstacles)
            gaps = np.where(~obstacle_map)[0]
            
            if len(gaps) == 0:
                return None
            
            # Find the longest continuous gap
            if len(gaps) > 0:
                # Split gaps into continuous segments
                gap_diffs = np.diff(gaps)
                gap_starts = np.concatenate(([0], np.where(gap_diffs > 1)[0] + 1))
                gap_ends = np.concatenate((np.where(gap_diffs > 1)[0] + 1, [len(gaps)]))
                
                # Find the largest gap
                gap_sizes = gap_ends - gap_starts
                largest_gap_idx = np.argmax(gap_sizes)
                
                if gap_sizes[largest_gap_idx] >= min_hole_size:
                    gap_start = gaps[gap_starts[largest_gap_idx]]
                    gap_end = gaps[gap_ends[largest_gap_idx] - 1]
                    hole_center_y = (gap_start + gap_end) / 2 * 0.01

                    return 0, hole_center_y
            
            return None
        else:
            return None
        