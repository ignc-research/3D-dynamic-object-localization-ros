#!/bin/bash
set -e

echo "ROS_DISTRO $ROS_DISTRO"
echo "ROS_ROOT   $ROS_ROOT"
PWD="$(pwd)"
ROS_ENV_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
ROS_WORKSPACE_SETUP="$PWD/devel/setup.bash"

echo "Source ROS_ENV_SETUP  $ROS_ENV_SETUP"
source "$ROS_ENV_SETUP"

echo "Source ROS_WORKSPACE_SETUP  $ROS_WORKSPACE_SETUP"
source "$ROS_WORKSPACE_SETUP"

# to start terminal 
exec "$@"