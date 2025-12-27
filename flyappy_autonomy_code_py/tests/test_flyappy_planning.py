import numpy as np
import pytest
from flyappy_autonomy_code.core.flyappy_planning import GapFinder

@pytest.fixture
def gapfinder():
    """Return a default GapFinder instance for tests."""
    return GapFinder(
        bird_pixel_size=2,
        slice_pixel_size=(0, 4)
        )

def test_all_free(gapfinder):
    """
    Scenario: 
        - All rows of map instance are free 
    
    Expectation:
        - The center row is selected as the best gap.
    """
    obstacle_map = np.zeros((4, 5), dtype=np.uint8)
    pos = (0, 2)
    x, y = gapfinder.find_free_rows(obstacle_map, pos)
    assert y == 2

def test_no_free_rows(gapfinder):
    """
    Scenario:
        - No free rows 
    Expectation: 
        - should fallback to current y position.
    """
    obstacle_map = np.full((4, 5), 255, dtype=np.uint8)
    pos = (0, 2)
    x, y = gapfinder.find_free_rows(obstacle_map, pos)
    assert y == pos[1]

def test_single_free_gap(gapfinder):
    """
    Scenario : 
        - Map with a single wide gap.
        - center of gap is explicitly not in the middle of the map.

    Expectation:
        - The center of the gap is selected.
    """
    obstacle_map = np.full((20, 20), 255, dtype=np.uint8)  # fully blocked
    
    obstacle_map[0:20, 8:12] = np.uint8(0) # create a gap

    print(obstacle_map)
    pos = (0, 4)  # current bird y position (column)
    x, y = gapfinder.find_free_rows(
        obstacle_map,
        pos)

    # Gap center is at column 10 (middle of 8-12)
    assert y in [9, 10, 11]
    # algorithm chooses closest to current position
    assert y == 9
