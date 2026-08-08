#!/usr/bin/env bash
set -euo pipefail

COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-${HOME}/opt/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
  fi
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ ! -x "${COPPELIASIM_ROOT}/coppeliaSim.sh" ]; then
  echo "CoppeliaSim installation not found: ${COPPELIASIM_ROOT}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}/platforms"
unset QT_DEBUG_PLUGINS

ARGS=()
for arg in "$@"; do
  case "${arg}" in
    *.ttt|*.ttm|*.simscene.xml|*.simmodel.xml)
      if [ -f "${arg}" ]; then
        ARGS+=("$(realpath "${arg}")")
      else
        ARGS+=("${arg}")
      fi
      ;;
    *)
      ARGS+=("${arg}")
      ;;
  esac
done

cd "${COPPELIASIM_ROOT}"
exec "${COPPELIASIM_ROOT}/coppeliaSim.sh" -G "python=${PYTHON_BIN}" "${ARGS[@]}"
