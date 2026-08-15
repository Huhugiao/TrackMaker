#!/usr/bin/env bash
# Single entry point for a reproducible CoppeliaSim + ROS 2 policy episode.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROS_ENV="${TRACKMAKER_ROS_ENV:-${HOME}/miniconda3/envs/ros2humble}"
COPPELIA_ROOT="${COPPELIASIM_ROOT:-${HOME}/opt/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04}"
SCENE="${TRACKMAKER_SCENE:-${PROJECT_ROOT}/coppeliasim/scenes/trackmaker_turtlebot4_v2_1_scene.ttt}"
SCENE_EXPLICIT=0
if [[ -n "${TRACKMAKER_SCENE:-}" ]]; then SCENE_EXPLICIT=1; fi
PROFILE="${TRACKMAKER_PROFILE:-${PROJECT_ROOT}/coppeliasim/profiles/trackmaker_turtlebot4_v2_1_prior.json}"
SEED=20260326
MAX_STEPS=449
OUTPUT_DIR=""
RECORD_BAG=1
RECORD_VIDEO=1
CAMERA_VIEW="overhead"
MODE="policy"

usage() {
    echo "Usage: $0 [--seed N] [--max-steps N] [--output-dir PATH] [--scene PATH] [--profile PATH] [--camera-view overhead|oblique] [--no-media|--no-bag|--no-video] [--calibrate|--interface-smoke]"
}

while (($#)); do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --scene) SCENE="$2"; SCENE_EXPLICIT=1; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --camera-view) CAMERA_VIEW="$2"; shift 2 ;;
        --no-media) RECORD_BAG=0; RECORD_VIDEO=0; shift ;;
        --no-bag) RECORD_BAG=0; shift ;;
        --no-video) RECORD_VIDEO=0; shift ;;
        --calibrate) MODE="calibration"; RECORD_VIDEO=0; shift ;;
        --interface-smoke) MODE="smoke"; RECORD_BAG=0; RECORD_VIDEO=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "${MODE}" == "calibration" && "${SCENE_EXPLICIT}" -eq 0 ]]; then
    SCENE="${PROJECT_ROOT}/coppeliasim/scenes/trackmaker_turtlebot4_v2_1_calibration_scene.ttt"
fi
MANIFEST="${SCENE%.*}.json"

if [[ "${CAMERA_VIEW}" != "overhead" && "${CAMERA_VIEW}" != "oblique" ]]; then
    echo "Unknown camera view: ${CAMERA_VIEW}" >&2
    exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/coppeliasim/ros2_demo/seed_${SEED}_$(date +%Y%m%d_%H%M%S)"
fi
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Refusing to overwrite existing output path: ${OUTPUT_DIR}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_DIR}"

for required in "${ROS_ENV}/setup.bash" "${COPPELIA_ROOT}/coppeliaSim.sh" "${SCENE}" "${MANIFEST}" "${PROFILE}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required deployment asset: ${required}" >&2
        exit 2
    fi
done

POLICY_PYTHON="${TRACKMAKER_POLICY_PYTHON:-$(dirname "${PROJECT_ROOT}")/bin/python}"
if [[ ! -x "${POLICY_PYTHON}" ]]; then
    echo "Missing lnenv Python interpreter: ${POLICY_PYTHON}" >&2
    exit 2
fi
PROFILE_CHECKSUM="$("${POLICY_PYTHON}" -c 'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from coppelia_env.digital_twin import load_profile; print(load_profile(Path(sys.argv[2]))["checksum"])' "${PROJECT_ROOT}" "${PROFILE}")"
MANIFEST_CHECKSUM="$("${POLICY_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["profile"]["checksum"])' "${MANIFEST}")"
if [[ "${MANIFEST_CHECKSUM}" != "${PROFILE_CHECKSUM}" ]]; then
    echo "Scene manifest/profile checksum mismatch: ${MANIFEST_CHECKSUM} != ${PROFILE_CHECKSUM}" >&2
    exit 2
fi
printf '%s\n' "${PROFILE_CHECKSUM}" > "${OUTPUT_DIR}/profile_checksum.txt"
cp "${PROFILE}" "${OUTPUT_DIR}/profile_input.json"

set +u
source "${ROS_ENV}/setup.bash"
set -u

if [[ "${MODE}" == "calibration" ]]; then
    # Two independent lanes in the flat calibration scene; no pose resets.
    SPAWN="-2.5,-1.8,0.0,-0.5,-0.7,0.0,2.2,2.2"
else
    SPAWN="$(python "${PROJECT_ROOT}/coppeliasim/tools/trackmaker_ros2_spawn.py" --seed "${SEED}" --manifest "${MANIFEST}")"
    python "${PROJECT_ROOT}/coppeliasim/tools/trackmaker_ros2_spawn.py" --seed "${SEED}" --manifest "${MANIFEST}" --json > "${OUTPUT_DIR}/spawn.json"
fi

COPPELIA_PID=""
BAG_PID=""
VIDEO_PID=""

stop_process_group() {
    local pid="$1"
    [[ -n "${pid}" ]] || return 0
    if kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null; then
        kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            if ! kill -0 -- "-${pid}" 2>/dev/null && ! kill -0 "${pid}" 2>/dev/null; then break; fi
            sleep 0.1
        done
        if kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null; then
            kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        fi
        for _ in $(seq 1 20); do
            if ! kill -0 -- "-${pid}" 2>/dev/null && ! kill -0 "${pid}" 2>/dev/null; then break; fi
            sleep 0.1
        done
        if kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null; then
            kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
        fi
    fi
    wait "${pid}" 2>/dev/null || true
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
    "-GtrackmakerSeed=${SEED}" \
    "-GtrackmakerProfileChecksum=${PROFILE_CHECKSUM}" \
    "-GtrackmakerCamera=${RECORD_VIDEO}" \
    "-GtrackmakerCameraView=${CAMERA_VIEW}" \
    -h -vinfo \
    "-a${PROJECT_ROOT}/coppeliasim/ros2/trackmaker_ros2_bridge.lua" \
    -s0 "${SCENE}")
# CoppeliaSim's client exits when stdin is not a terminal. The tiny PTY wrapper
# keeps stdin open without creating a second process group.
READY=0
FATAL_STARTUP=0
for ATTEMPT in 1 2 3; do
    echo "CoppeliaSim startup attempt ${ATTEMPT}" >> "${OUTPUT_DIR}/coppeliasim.log"
    setsid python "${PROJECT_ROOT}/coppeliasim/tools/pty_exec.py" "${COPPELIA_COMMAND[@]}" \
        >> "${OUTPUT_DIR}/coppeliasim.log" 2>&1 &
    COPPELIA_PID=$!
    for CHECK in $(seq 1 60); do
        if timeout 1 ros2 topic echo --once /clock >/dev/null 2>&1 && \
           ros2 topic list 2>/dev/null | rg -qx '/tracking/defender/pose' && \
           ros2 topic list 2>/dev/null | rg -qx '/demo/profile_metadata' && \
           ros2 topic list 2>/dev/null | rg -qx '/defender/joint_states' && \
           { [[ "${RECORD_VIDEO}" -eq 0 ]] || ros2 topic list 2>/dev/null | rg -qx '/demo/camera/image_raw'; }; then
            READY=1
            break
        fi
        # The PTY/Xvfb wrapper can spend more than a second loading a large
        # scene before the final binary is visible by its exact process name.
        # The process-group leader is the stable liveness contract.
        if [[ "${CHECK}" -gt 4 ]] && ! kill -0 "${COPPELIA_PID}" 2>/dev/null; then
            break
        fi
        if rg -q 'runtime introspection failed|trackmaker_ros2_bridge@addOnScript:error|ROS 2 V2\.1 bridge@addOnScript:error' "${OUTPUT_DIR}/coppeliasim.log"; then
            FATAL_STARTUP=1
            break
        fi
        sleep 0.25
    done
    if [[ "${READY}" -eq 1 ]]; then break; fi
    stop_process_group "${COPPELIA_PID}"
    COPPELIA_PID=""
    if [[ "${FATAL_STARTUP}" -eq 1 ]]; then break; fi
    sleep 1
done
if [[ "${READY}" -ne 1 ]]; then
    echo "CoppeliaSim failed to publish live ROS 2 topics after three attempts; see ${OUTPUT_DIR}/coppeliasim.log" >&2
    exit 3
fi

if [[ "${RECORD_BAG}" -eq 1 ]]; then
    setsid ros2 bag record \
        -o "${OUTPUT_DIR}/episode_bag" \
        /clock /tf \
        /tracking/defender/pose /tracking/attacker/pose /tracking/target/pose \
        /defender/scan /attacker/scan \
        /defender/cmd_vel /attacker/cmd_vel \
        /defender/joint_targets /attacker/joint_targets \
        /defender/joint_states /attacker/joint_states \
        /defender/actuator_events /attacker/actuator_events \
        /defender/actuator_state /attacker/actuator_state \
        /defender/collision /attacker/collision \
        /demo/profile_metadata /demo/selected_skill /demo/outcome /diagnostics \
        > "${OUTPUT_DIR}/rosbag.log" 2>&1 &
    BAG_PID=$!
fi

if [[ "${RECORD_VIDEO}" -eq 1 ]]; then
    setsid python "${PROJECT_ROOT}/coppeliasim/tools/trackmaker_ros2_video.py" \
        --output "${OUTPUT_DIR}/episode.mp4" > "${OUTPUT_DIR}/video.log" 2>&1 &
    VIDEO_PID=$!
fi

set +e
if [[ "${MODE}" == "calibration" ]]; then
    python "${PROJECT_ROOT}/coppeliasim/tools/trackmaker_ros2_motion_calibration.py" \
        --output "${OUTPUT_DIR}/motion_calibration.json" \
        --profile "${PROFILE}" \
        2>&1 | tee "${OUTPUT_DIR}/calibration.log"
elif [[ "${MODE}" == "smoke" ]]; then
    python "${PROJECT_ROOT}/coppeliasim/tools/trackmaker_ros2_interface_smoke.py" \
        --duration-s 3.0 \
        --output "${OUTPUT_DIR}/interface_smoke.json" \
        2>&1 | tee "${OUTPUT_DIR}/interface_smoke.log"
else
    python "${PROJECT_ROOT}/coppeliasim/tools/trackmaker_ros2_policy.py" \
        --seed "${SEED}" \
        --max-steps "${MAX_STEPS}" \
        --manifest "${MANIFEST}" \
        --output-json "${OUTPUT_DIR}/episode.json" \
        2>&1 | tee "${OUTPUT_DIR}/policy.log"
fi
POLICY_CODE=${PIPESTATUS[0]}
set -e

if [[ "${RECORD_VIDEO}" -eq 1 ]]; then
    for _ in $(seq 1 50); do
        if ! kill -0 "${VIDEO_PID}" 2>/dev/null; then break; fi
        sleep 0.2
    done
    stop_process_group "${VIDEO_PID}"
    VIDEO_PID=""
fi
if [[ "${RECORD_BAG}" -eq 1 ]]; then
    stop_process_group "${BAG_PID}"
    BAG_PID=""
fi
stop_process_group "${COPPELIA_PID}"
COPPELIA_PID=""

if [[ "${MODE}" == "calibration" && "${POLICY_CODE}" -eq 0 ]]; then
    set +e
    env -u PYTHONPATH -u PYTHONHOME "${POLICY_PYTHON}" "${PROJECT_ROOT}/coppeliasim/tools/trackmaker_fit_digital_twin.py" \
        --dataset "${OUTPUT_DIR}/motion_calibration.json" \
        --prior-profile "${PROFILE}" \
        --output-profile "${OUTPUT_DIR}/measured_profile.json" \
        --output-report "${OUTPUT_DIR}/fit_report.json" \
        2>&1 | tee "${OUTPUT_DIR}/fit.log"
    FIT_CODE=${PIPESTATUS[0]}
    set -e
    if [[ "${FIT_CODE}" -ne 0 ]]; then POLICY_CODE="${FIT_CODE}"; fi
fi

if [[ "${POLICY_CODE}" -ne 0 ]]; then
    echo "TrackMaker ${MODE} failed with code ${POLICY_CODE}; see ${OUTPUT_DIR} logs" >&2
    exit "${POLICY_CODE}"
fi
RESULT_JSON="${OUTPUT_DIR}/episode.json"
if [[ "${MODE}" == "calibration" ]]; then RESULT_JSON="${OUTPUT_DIR}/motion_calibration.json"; fi
if [[ "${MODE}" == "smoke" ]]; then RESULT_JSON="${OUTPUT_DIR}/interface_smoke.json"; fi
if [[ ! -s "${RESULT_JSON}" ]]; then
    echo "Episode completed but JSON output is missing" >&2
    exit 4
fi

if [[ "${RECORD_BAG}" -eq 1 && ! -f "${OUTPUT_DIR}/episode_bag/metadata.yaml" ]]; then
    echo "Episode completed but rosbag output is missing" >&2
    exit 4
fi
if [[ "${RECORD_VIDEO}" -eq 1 ]]; then
    if [[ ! -s "${OUTPUT_DIR}/episode.mp4" ]]; then
        echo "Episode completed but MP4 output is missing" >&2
        exit 4
    fi
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=codec_name,width,height,pix_fmt \
        -of default=noprint_wrappers=1 "${OUTPUT_DIR}/episode.mp4"
fi
echo "TrackMaker ROS 2 ${MODE} complete: ${OUTPUT_DIR}"
