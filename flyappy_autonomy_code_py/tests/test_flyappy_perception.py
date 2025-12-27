import sys
from unittest.mock import MagicMock

# Fake sensor_msgs.msg.LaserScan
sys.modules['sensor_msgs'] = MagicMock()
sys.modules['sensor_msgs.msg'] = MagicMock()

from flyappy_autonomy_code.core.flyappy_perception import PositionEstimator, OccupancyGridMapper

import numpy as np
import pytest


class MockLaserScan:
    """Mock of sensor_msgs.msg.LaserScan."""
    def __init__(
        self,
        ranges,
        intensities,
        angle_min=0.0,
        angle_increment=0.0,
        angle_max=0.0,
        time_increment=0.0,
        scan_time=0.0,
        range_min=0.0,
        range_max=10.0,
        header=None
    ):
        self.header = header
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = angle_increment
        self.time_increment = time_increment
        self.scan_time = scan_time
        self.range_min = range_min
        self.range_max = range_max
        self.ranges = ranges
        self.intensities = intensities

@pytest.fixture
def simple_scan():
    """Fixture for a simple 1-beam laser scan."""
    return MockLaserScan(
        ranges=[2.0],
        intensities=[1.0],
        angle_min=0.0,
        angle_increment=0.0
    )

@pytest.fixture
def mapper():
    """Fixture for a fresh OccupancyGridMapper."""
    m = OccupancyGridMapper(map_size=(5, 5), map_resolution=1.0, agent_size=1)
    m.reset()
    return m


def test_map_roll():
    """
    Scenario:
        - An known occupancy grid is set
        - the agent's position is set to 2.7
         
    Expectation:
        - After rolling, the map is shifted correctly
        - New areas are filled with unknown value (127)
        - The map origin is updated correctly, meaning by the floored
          value of the agent's position.
    """
    mapper = OccupancyGridMapper(map_size=(5, 5), map_resolution=1.0, agent_size=1)
    mapper.obstacle_map = np.arange(25).reshape((5, 5))

    mapper.map_roll((2.7, 0.0)) # roll by floor value of 2.7 (2 cells)

    expected_map = np.array([
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [127, 127, 127, 127, 127],
        [127, 127, 127, 127, 127]

    ])
    assert np.array_equal(mapper.obstacle_map, expected_map)
    assert mapper.map_origin == (2, 0)

def test_map_add_scan(simple_scan, mapper):
    """
    Scenario:
        - A simple 1-beam laser scan is added to the map
        - The agent is at position (0.0, 2.0)

    Expectation:
        - The map is updated correctly with occupied and free cells
        - Unknown cells remain unchanged.
        - The occupied cell is at (3,2) and free cells are at (1,2) and (2,2)
    """
    mapper.map_add_scan(
        laser_scan=simple_scan,
        position=(0.0, 2.0),
    )

    expected_map = np.array([
        [127, 127, 127, 127, 127],
        [127, 127,   0, 127, 127],
        [127, 127,   0, 127, 127],
        [127, 127, 255, 127, 127],
        [127, 127, 127, 127, 127]
    ])
    print(mapper.obstacle_map)
    assert np.array_equal(mapper.obstacle_map, expected_map)





@pytest.fixture
def position_estimator():
    """Fresh PositionEstimator for each test."""
    return PositionEstimator(
        initial_position=(0.0, 1.5)
    )


def test_update_position_basic(position_estimator):
    """
    Scenario:
        - The agent starts at (0.0, 0.0)
        - A velocity of (1.0, -0.5) is applied for 2.0 seconds
    Expectation:
        - The new position is (2.0, -1.0)
    """
    
    position_estimator.update_position(
        current_velocity=(1.0, -0.5),
        timestep=2.0
    )
    assert position_estimator.get_position() == (2.0, 0.5)

def test_set_position(position_estimator):
    """set_position overrides current position."""
    position_estimator.set_position((3.2, -4.1))
    assert position_estimator.get_position() == (3.2, -4.1)
