#!/usr/bin/env bash
# Single entry point for a reproducible CoppeliaSim + ROS 2 policy episode.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROS_ENV="${TRACKMAKER_ROS_ENV:-${HOME}/miniconda3/envs/ros2humble}"
COPPELIA_ROOT="${COPPELIASIM_ROOT:-${HOME}/opt/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04}"
SCENE="${TRACKMAKER_SCENE:-${PROJECT_ROOT}/coppeliasim/scenes/trackmaker_turtlebot4_scene.ttt}"
SEED=20260326
MAX_STEPS=449
OUTPUT_DIR=""
RECORD_MEDIA=1
CAMERA_VIEW="overhead"
MODE="policy"

usage() {
    echo "Usage: $0 [--seed N] [--max-steps N] [--output-dir PATH] [--camera-view overhead|oblique] [--no-media] [--calibrate]"
}

while (($#)); do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --camera-view) CAMERA_VIEW="$2"; shift 2 ;;
        --no-media) RECORD_MEDIA=0; shift ;;
        --calibrate) MODE="calibration"; RECORD_MEDIA=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "${CAMERA_VIEW}" != "overhead" && "${CAMERA_VIEW}" != "oblique" ]]; then
    echo "Unknown camera view: ${CAMERA_VIEW}" >&2
    exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/coppeliasim/ros2_demo/seed_${SEED}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(realpath "${OUTPUT_DIR}")"

for required in "${ROS_ENV}/setup.bash" "${COPPELIA_ROOT}/coppeliaSim.sh" "${SCENE}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required deployment asset: ${required}" >&2
        exit 2
    fi
done

set +u
source "${ROS_ENV}/setup.bash"
set -u

if [[ "${MODE}" == "calibration" ]]; then
    SPAWN="-2.1,0.7,0.0,0.6,0.7,0.0,2.2,-2.0"
else
    SPAWN="$(python "${PROJECT_ROOT}/coppeliasim/tools/trackmaker_ros2_spawn.py" --seed "${SEED}")"
    python "${PROJECT_ROOT}/coppeliasim/tools/trackmaker_ros2_spawn.py" --seed "${SEED}" --json > "${OUTPUT_DIR}/spawn.json"
fi

COPPELIA_PID=""
BAG_PID=""
VIDEO_PID=""

stop_process_group() {
    local pid="$1"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
}

cleanup() {
    stop_process_group "${VIDEO_PID}"
    stop_process_group "${BAG_PID}"
    stop_process_group "${COPPELIA_PID}"
}
trap cleanup EXIT INT TERM

COPPELIA_COMMAND=(env \
    LD_LIBRARY_PATH="${ROS_ENV}/lib:${COPPELIA_ROOT}:${LD_LIBRARY_PATH:-}" \
    LD_PRELOAD="${ROS_ENV}/lib/libpython3.10.so" \
    xvfb-run -a -s '-screen 0 1920x1080x24' \
    "${COPPELIA_ROOT}/coppeliaSim.sh" \
    "-Gpython=${ROS_ENV}/bin/python" \
    -GpreferredSandboxLang=lua \
    "-GtrackmakerSpawn=${SPAWN}" \
    "-GtrackmakerCamera=${RECORD_MEDIA}" \
    "-GtrackmakerCameraView=${CAMERA_VIEW}" \
    -h -vinfo \
    "-a${PROJECT_ROOT}/coppeliasim/ros2/trackmaker_ros2_bridge.lua" \
    -s0 "${SCENE}")
# CoppeliaSim's client exits when stdin is not a terminal. The tiny PTY wrapper
# keeps stdin open without creating a second process group.
READY=0
for ATTEMPT in 1 2 3; do
    echo "CoppeliaSim startup attempt ${ATTEMPT}" >> "${OUTPUT_DIR}/coppeliasim.log"
    setsid python "${PROJECT_ROOT}/coppeliasim/tools/pty_exec.py" "${COPPELIA_COMMAND[@]}" \
        >> "${OUTPUT_DIR}/coppeliasim.log" 2>&1 &
    COPPELIA_PID=$!
    for CHECK in $(seq 1 60); do
        if timeout 1 ros2 topic echo --once /clock >/dev/null 2>&1 && \
           ros2 topic list 2>/dev/null | rg -qx '/tracking/defender/pose' && \
           { [[ "${RECORD_MEDIA}" -eq 0 ]] || ros2 topic list 2>/dev/null | rg -qx '/demo/camera/image_raw'; }; then
            READY=1
            break
        fi
        if [[ "${CHECK}" -gt 4 ]] && ! pgrep -g "${COPPELIA_PID}" -x coppeliaSim >/dev/null 2>&1; then
            break
        fi
        sleep 0.25
    done
    if [[ "${READY}" -eq 1 ]]; then break; fi
    stop_process_group "${COPPELIA_PID}"
    COPPELIA_PID=""
    sleep 1
done
if [[ "${READY}" -ne 1 ]]; then
    echo "CoppeliaSim failed to publish live ROS 2 topics after three attempts; see ${OUTPUT_DIR}/coppeliasim.log" >&2
    exit 3
fi

if [[ "${RECORD_MEDIA}" -eq 1 ]]; then
    setsid ros2 bag record \
        -o "${OUTPUT_DIR}/episode_bag" \
        /clock /tf \
        /tracking/defender/pose /tracking/attacker/pose /tracking/target/pose \
        /defender/scan /attacker/scan \
        /defender/cmd_vel /attacker/cmd_vel \
        /defender/collision /attacker/collision \
        /demo/selected_skill /demo/outcome /diagnostics \
        > "${OUTPUT_DIR}/rosbag.log" 2>&1 &
    BAG_PID=$!

    setsid python "${PROJECT_ROOT}/coppeliasim/tools/trackmaker_ros2_video.py" \
        --output "${OUTPUT_DIR}/episode.mp4" > "${OUTPUT_DIR}/video.log" 2>&1 &
    VIDEO_PID=$!
fi

set +e
if [[ "${MODE}" == "calibration" ]]; then
    python "${PROJECT_ROOT}/coppeliasim/tools/trackmaker_ros2_motion_calibration.py" \
        --output "${OUTPUT_DIR}/motion_calibration.json" \
        2>&1 | tee "${OUTPUT_DIR}/calibration.log"
else
    python "${PROJECT_ROOT}/coppeliasim/tools/trackmaker_ros2_policy.py" \
        --seed "${SEED}" \
        --max-steps "${MAX_STEPS}" \
        --output-json "${OUTPUT_DIR}/episode.json" \
        2>&1 | tee "${OUTPUT_DIR}/policy.log"
fi
POLICY_CODE=${PIPESTATUS[0]}
set -e

if [[ "${RECORD_MEDIA}" -eq 1 ]]; then
    for _ in $(seq 1 50); do
        if ! kill -0 "${VIDEO_PID}" 2>/dev/null; then break; fi
        sleep 0.2
    done
    stop_process_group "${VIDEO_PID}"
    VIDEO_PID=""
    stop_process_group "${BAG_PID}"
    BAG_PID=""
fi
stop_process_group "${COPPELIA_PID}"
COPPELIA_PID=""

if [[ "${POLICY_CODE}" -ne 0 ]]; then
    echo "TrackMaker ${MODE} failed with code ${POLICY_CODE}; see ${OUTPUT_DIR} logs" >&2
    exit "${POLICY_CODE}"
fi
RESULT_JSON="${OUTPUT_DIR}/episode.json"
if [[ "${MODE}" == "calibration" ]]; then RESULT_JSON="${OUTPUT_DIR}/motion_calibration.json"; fi
if [[ ! -s "${RESULT_JSON}" ]]; then
    echo "Episode completed but JSON output is missing" >&2
    exit 4
fi

if [[ "${RECORD_MEDIA}" -eq 1 ]]; then
    if [[ ! -s "${OUTPUT_DIR}/episode.mp4" || ! -f "${OUTPUT_DIR}/episode_bag/metadata.yaml" ]]; then
        echo "Episode completed but MP4 or rosbag output is missing" >&2
        exit 4
    fi
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=codec_name,width,height,pix_fmt \
        -of default=noprint_wrappers=1 "${OUTPUT_DIR}/episode.mp4"
fi
echo "TrackMaker ROS 2 ${MODE} complete: ${OUTPUT_DIR}"
