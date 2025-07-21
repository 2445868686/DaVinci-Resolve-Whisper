#!/usr/bin/env bash
set -euo pipefail
# 获取管理员权限
if [ "$EUID" -ne 0 ]; then
  echo "Enter Password："
  exec sudo "$0" "$@"
  exit
fi
# ———————— 配置变量 ————————
# Python 解释器路径
PYTHON=python3

# 将所有需要安装的包定义在一个数组中
# 新增了 setuptools 和 wheel，它们是从源码（如 .tar.gz）构建和安装包所必需的工具
# pip 会智能地利用它们来处理 googletrans 的源码包
PACKAGES=(
  "faster_whisper==1.1.1"
)

# DaVinci Resolve Fusion 脚本目录
WHEEL_DIR="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/HB/Whisper/wheel"
TARGET_DIR="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/HB/Whisper/Lib"

# 使用的 PyPI 镜像源
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# ———————— 日志函数 ————————
log() {
  local level="$1"; shift
  local msg="$*"
  local ts
  ts=$(date +"%Y-%m-%d %H:%M:%S")
  # 为不同的日志级别添加颜色
  case "$level" in
    INFO)    echo -e "[$ts][\033[34mINFO\033[0m]    $msg" ;;
    SUCCESS) echo -e "[$ts][\033[32mSUCCESS\033[0m] $msg" ;;
    WARN)    echo -e "[$ts][\033[33mWARN\033[0m]    $msg" ;;
    ERROR)   echo -e "[$ts][\033[91mERROR\033[0m]   $msg" ;;
    *)       echo "[$ts][$level] $msg" ;;
  esac
}

# ———————— 脚本开始 ————————
log INFO "Starting unified offline download & installation."
log INFO "Packages: ${PACKAGES[*]}"
echo "------------------------------------------------------------"

# ———————— 步骤 1：准备目录和权限 ————————
log INFO "[1/4] Preparing directories and permissions..."
mkdir -p "$WHEEL_DIR"
# 目标目录通常需要 sudo 权限来创建和修改
# 我们创建它，然后把所有权交给当前用户，这样后续 pip 操作就不再需要 sudo
sudo mkdir -p "$TARGET_DIR"
sudo chown -R "$(whoami)" "$TARGET_DIR"
log SUCCESS "Directories are ready. Target owned by $(whoami)."

# ———————— 步骤 2：清理 pip 缓存 (可选) ————————
log INFO "[2/4] Clearing pip cache..."
$PYTHON -m pip cache purge >/dev/null 2>&1 || log WARN "Failed to purge pip cache or cache was already empty."

# ———————— 步骤 3：从官方源尝试下载，如失败再使用镜像源 ————————
log INFO "[3/4] Attempting to download all packages from official PyPI..."
if $PYTHON -m pip download "${PACKAGES[@]}" \
    --dest "$WHEEL_DIR" \
    --no-cache-dir \
    -i https://pypi.org/simple; then
  log SUCCESS "Packages successfully downloaded from official PyPI to: $WHEEL_DIR"
else
  log WARN "Official PyPI download failed. Trying mirror source: $PIP_MIRROR ..."
  if $PYTHON -m pip download "${PACKAGES[@]}" \
      --dest "$WHEEL_DIR" \
      --no-cache-dir \
      -i "$PIP_MIRROR"; then
    log SUCCESS "Packages successfully downloaded from mirror to: $WHEEL_DIR"
  else
    log ERROR "Download failed from both official and mirror sources. Check network or mirror availability."
    exit 1
  fi
fi

# ———————— 步骤 4：从本地目录离线安装所有包 ————————
log INFO "[4/4] Offline installing all packages to: $TARGET_DIR"
if $PYTHON -m pip install "${PACKAGES[@]}" \
    --no-index \
    --find-links "$WHEEL_DIR" \
    --target "$TARGET_DIR" \
    --disable-pip-version-check \
    --upgrade; then
  log SUCCESS "All packages installed successfully!"
else
  log ERROR "Offline installation failed. Check the logs above."
  exit 1
fi

# ———————— 步骤 5：完成 ————————
echo "------------------------------------------------------------"
log SUCCESS "Installation complete. Modules are located in: $TARGET_DIR"
log SUCCESS "All done 🎉"