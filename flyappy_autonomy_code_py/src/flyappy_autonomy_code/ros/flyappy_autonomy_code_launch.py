from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
import sys
import os


def _parse_rviz_display(argv):
	"""Extract rviz_display value from argv."""
	for arg in argv:
		if arg.startswith("rviz_display:="):
			return arg.split(":=")[1].lower() in ("true", "1", "yes")
	return True  # default to True


def generate_launch_description() -> LaunchDescription:
	"""Launch for the Flyappy autonomy code
     
    Launched nodes:
        - perception_node
        - planning_node
        - control_node.
        - rviz2 (if rviz_display is True)
    """
	
	# Parse rviz_display from CLI args
	rviz_display = _parse_rviz_display(sys.argv[1:])

	control_rate_arg = DeclareLaunchArgument(
		"control_rate",
		default_value="30.0",
		description="Control loop frequency (Hz)",
	)
	perception_rate_arg = DeclareLaunchArgument(
		"perception_rate",
		default_value="30.0",
		description="Perception loop frequency (Hz)",
	)
	planning_rate_arg = DeclareLaunchArgument(
		"planning_rate",
		default_value="30.0",
		description="Planning loop frequency (Hz)",
	)
	rviz_display_arg = DeclareLaunchArgument(
		"rviz_display",
		default_value="True",
		description="Opens rviz2 with config to visualize OccupancyGrid",
	)

	control_node = ExecuteProcess(
		cmd=["control_node"],
		output="screen",
	)

	perception_node = ExecuteProcess(
		cmd=["perception_node"],
		output="screen",
	)

	planning_node = ExecuteProcess(
		cmd=["planning_node"],
		output="screen",
	)

	# Only add RViz2 and rqt if debug_mode is enabled
	nodes = [
		control_rate_arg,
		perception_rate_arg,
		planning_rate_arg,
		rviz_display_arg,
		control_node,
		perception_node,
		planning_node,
	]

	if rviz_display:
		rviz_config_path = os.path.join(os.path.dirname(__file__), "rviz", "flyappy_viewer.rviz")
		rviz_node = ExecuteProcess(
			cmd=["rviz2", "-d", rviz_config_path],
			output="screen",
		)
		nodes.append(rviz_node)

	return LaunchDescription(nodes)


def main() -> None:
    from launch.launch_service import LaunchService
    ls = LaunchService(argv=sys.argv[1:])
    ls.include_launch_description(generate_launch_description())
    ls.run()


if __name__ == "__main__":
    main()