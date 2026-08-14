#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROS_ENV="${TRACKMAKER_ROS_ENV:-${HOME}/miniconda3/envs/ros2humble}"
COPPELIA_ROOT="${COPPELIASIM_ROOT:-${HOME}/opt/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04}"
WORKSPACE="${TRACKMAKER_SIMROS2_WS:-${PROJECT_ROOT}/outputs/coppeliasim/simros2_ws}"
SOURCE="${COPPELIA_ROOT}/programming/ros2_packages/sim_ros2_interface"
PACKAGE="${WORKSPACE}/src/sim_ros2_interface"

if [[ ! -f "${ROS_ENV}/setup.bash" ]]; then
  echo "ROS 2 environment not found: ${ROS_ENV}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE}/package.xml" ]]; then
  echo "CoppeliaSim simROS2 source not found: ${SOURCE}" >&2
  exit 2
fi

# shellcheck disable=SC1090
set +u
source "${ROS_ENV}/setup.bash"
set -u
export PATH="${ROS_ENV}/bin:${PATH}"
export COPPELIASIM_ROOT_DIR="${COPPELIA_ROOT}"
mkdir -p "${WORKSPACE}/src"
rm -rf "${PACKAGE}"
rm -rf "${WORKSPACE}/build/sim_ros2_interface" "${WORKSPACE}/install/sim_ros2_interface"
cp -a "${SOURCE}" "${PACKAGE}"

for interface in geometry_msgs/msg/PoseStamped rosgraph_msgs/msg/Clock; do
  if ! grep -qxF "${interface}" "${PACKAGE}/meta/interfaces.txt"; then
    sed -i "\$a${interface}" "${PACKAGE}/meta/interfaces.txt"
  fi
done

# coppeliasim_add_plugin installs into COPPELIASIM_ROOT as part of its build.
if [[ ! -f "${COPPELIA_ROOT}/libsimROS2.so.trackmaker-original" ]]; then
  cp -a "${COPPELIA_ROOT}/libsimROS2.so" "${COPPELIA_ROOT}/libsimROS2.so.trackmaker-original"
fi

colcon --log-base "${WORKSPACE}/log" build \
  --base-paths "${PACKAGE}" \
  --build-base "${WORKSPACE}/build" \
  --install-base "${WORKSPACE}/install" \
  --cmake-args -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="${ROS_ENV}/bin/python"

BUILT_LIBRARY="${WORKSPACE}/build/sim_ros2_interface/libsimROS2.so"
if [[ ! -f "${BUILT_LIBRARY}" ]]; then
  echo "simROS2 build completed without ${BUILT_LIBRARY}" >&2
  exit 3
fi
cp -a "${BUILT_LIBRARY}" "${COPPELIA_ROOT}/libsimROS2.so"
echo "installed TrackMaker simROS2 plugin: ${COPPELIA_ROOT}/libsimROS2.so"
