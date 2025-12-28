# flyappy_autonomy_code_py

## Description

This repository contains the code automating the Flyappy agent using Python. It is structured as a Python package exposing multiple nodes for perception, planning, and control aswell as 2 comprehensive entry points. The autonomy code processes sensor data, builds an occupancy grid map, finds gaps, and sends control commands to navigate the environment.

## Project Structure

```bash
flyappy_autonomy_code_py/
    src/flyappy_autonomy_code/
        core/
            flyappy_control.py
            flyappy_perception.py
            flyappy_planning.py
        ros/
            control_node.py
            perception_node.py
            planning_node.py
            flyappy_autonomy_code_node.py
            flyappy_autonomy_code_launch.py
```

## Architecture

- **`flyappy_perception.py`**:
  - `PositionEstimator`: Estimates agent position from odometry
  - `OccupancyGridMapper`: Builds 2D OccupancyGrid from LaserScan and odometry data
  
- **`flyappy_planning.py`**:
  - `GapFinder`: Detects gaps in OccupancyGrid

- **`flyappy_control.py`**:
  - `ProportionalController`: Computes velocity commands to reach targets
  - `StateFeedbackController`: Combines perception and planning to control the agent

## Prerequisites

- ROS 2 Jazzy (desktop install recommended for rviz2)
- Python 3.12
- numpy<1.28>

## Setup

Use the virtual environment and install the package as explained in the global repository README.

Activate the python virtual environment:

```bash
# From repo root
source .venv/bin/activate
```

Install the main game:

```bash
# From repo root (-e for editable/development mode)
pip install -e ./flyappy_main_game
```

## Running

Console scripts are exposed via `pyproject.toml`. After installation, a few options are available.

### Quick start

The recommended way to launch the full autonomy nodes along with rviz visualization is through the provided launch script:

```bash
flyappy_autonomy_code_launch 
```

When launched this way alongside the `flyappy_main_game`, the autonomy notes will take over control of the agent a soon as you start the game by pressing the prompted keys. A window with rviz2 should also open, displaying the OccupancyGrid being built as the agent advances. Once finished, you may restart the game by pressing the prompted keys or shutdown the node by pressing `Ctrl+C`.

Should you not want to launch rviz to visualize the OccupancyGrid at runtime use:

```bash
flyappy_autonomy_code_launch rviz_display:=False
```

### Advanced

For development or debugging, you may want to run individual nodes or adjust their parameters.

```bash
# Individual nodes
control_node
perception_node
planning_node
```

```bash
# Adjust parameters (e.g., control rate)
control_node --ros-args -p control_rate:=<desired_rate>
perception_node --ros-args -p perception_rate:=<desired_rate>
planning_node --ros-args -p planning_rate:=<desired_rate>
```

As an alternative to running multiple nodes, you can also run the all-in-one node:

```bash
# all-in-one node
flyappy_autonomy_code_node
```

Note: These scripts expect a ROS 2 environment (`rclpy`) and standard message packages to be available at runtime.

## Testing

Tests live under `tests/` and use `pytest`.

Installed package (simplest):

```bash
cd flyappy_autonomy_code_py
pytest -v
```

Without installation (use `PYTHONPATH`):

```bash
cd flyappy_autonomy_code_py
PYTHONPATH="$(pwd)/src" pytest -v
```

### ROS imports in tests

The test suite is designed to run without a full ROS environment by mocking message packages (e.g., `sensor_msgs`, `nav_msgs`, `geometry_msgs`).

```python
import sys
from unittest.mock import MagicMock

for pkg in (
        "sensor_msgs",
        "sensor_msgs.msg",
        "nav_msgs",
        "nav_msgs.msg",
        "geometry_msgs",
        "geometry_msgs.msg",
):
        sys.modules[pkg] = MagicMock()
```

## Troubleshooting

- **ImportError for ROS messages during tests:** ensure the mocking snippet is present before importing `core` modules that reference ROS message types, or run with the package installed and proper ROS env.

- **`ModuleNotFoundError` when running tests** set `PYTHONPATH=$(pwd)/src` inside `flyappy_autonomy_code_py`.

```bash
# From repo root
PYTHONPATH=$(pwd)/flyappy_autonomy_code_py/src pytest -p no:launch_testing flyappy_autonomy_code_py/tests/test_flyappy_control.py -v
```

- **NumPy version conflicts:** the package pins `numpy<1.28`. If you see ABI or import errors, try `pip install "numpy<1.28" --force-reinstall`.

## Author

ALLONAS Alexandre - Coding challenge for Flyability internship (2025)
