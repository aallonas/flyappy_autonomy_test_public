from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
	"""Launch for the Flyappy autonomy code
     
    Launched nodes:
        - perception_node
        - planning_node
        - control_node.
        - rviz2 (if debug_mode is True)
        - rqt_reconfigure (if debug_mode is True)
"""

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
	debug_mode_arg = DeclareLaunchArgument(
		"debug_mode",
		default_value="True",
		description="Enable perception debug visuals/logging",
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

	rviz_node = ExecuteProcess(
		cmd=["rviz2"],
		output="screen",
		condition=IfCondition(LaunchConfiguration("debug_mode")),
	)

	rqt_reconfigure_node = ExecuteProcess(
		cmd=["ros2", "run", "rqt_reconfigure", "rqt_reconfigure"],
		output="screen",
		condition=IfCondition(LaunchConfiguration("debug_mode")),
	)

	return LaunchDescription(
		[
			control_rate_arg,
			perception_rate_arg,
			planning_rate_arg,
			debug_mode_arg,
			control_node,
			perception_node,
			planning_node,
			rviz_node,
			rqt_reconfigure_node,
		]
	)

def main() -> None:
    from launch.launch_service import LaunchService
    ls = LaunchService(argv=[])
    ls.include_launch_description(generate_launch_description())
    ls.run()


if __name__ == "__main__":
    main()