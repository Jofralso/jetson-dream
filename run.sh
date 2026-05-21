#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Jetson-Dream: Quick Start Runner
# ═══════════════════════════════════════════════════════════════
# Detects hardware, downloads models if needed, launches with
# optimal settings. Run from project root:
#
#   ./run.sh              # Auto-detect hardware, launch
#   ./run.sh --nano      # Force Jetson Nano mode
#   ./run.sh --lite      # Use MobileNetV2 DeepDream
#   ./run.sh --docker    # Run via Docker
#   ./run.sh --test      # Run test suite
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────

detect_platform() {
    if [ -f /proc/device-tree/model ]; then
        MODEL=$(tr -d '\0' < /proc/device-tree/model)
        if echo "$MODEL" | grep -qi "jetson nano"; then
            echo "jetson-nano"
            return
        elif echo "$MODEL" | grep -qi "jetson"; then
            echo "jetson-orin"
            return
        fi
    fi
    if command -v nvidia-smi &>/dev/null; then
        echo "desktop-gpu"
    else
        echo "cpu-only"
    fi
}

PLATFORM=$(detect_platform)

# ─────────────────────────────────────────────────────────────
# Dependency checks
# ─────────────────────────────────────────────────────────────

check_python() {
    if ! command -v python3 &>/dev/null; then
        error "Python 3 not found. Install python3."
        exit 1
    fi
    ok "Python $(python3 --version 2>&1 | awk '{print $2}')"
}

check_torch() {
    if python3 -c "import torch" 2>/dev/null; then
        local ver
        ver=$(python3 -c "import torch; print(torch.__version__)")
        local cuda
        cuda=$(python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')")
        ok "PyTorch $ver ($cuda)"
        return 0
    else
        warn "PyTorch not found"
        return 1
    fi
}

check_opencv() {
    if python3 -c "import cv2" 2>/dev/null; then
        local ver
        ver=$(python3 -c "import cv2; print(cv2.__version__)")
        ok "OpenCV $ver"
        return 0
    else
        warn "OpenCV not found"
        return 1
    fi
}

check_camera() {
    if [ -e /dev/video0 ]; then
        ok "Camera /dev/video0"
        return 0
    else
        warn "No camera at /dev/video0 (will use test pattern or --stream only)"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────
# Model download
# ─────────────────────────────────────────────────────────────

ensure_models() {
    local model_count
    model_count=$(find models/ -name "*.pth" 2>/dev/null | wc -l)
    if [ "$model_count" -ge 4 ]; then
        ok "$model_count style models found"
    else
        info "Downloading style models..."
        python3 -m scripts.download_styles -o models --subset core
    fi
}

# ─────────────────────────────────────────────────────────────
# Install missing deps
# ─────────────────────────────────────────────────────────────

install_deps() {
    info "Installing Python dependencies..."
    pip3 install --quiet -r requirements.txt
    ok "Dependencies installed"
}

# ─────────────────────────────────────────────────────────────
# Docker mode
# ─────────────────────────────────────────────────────────────

run_docker() {
    local profile
    case "$PLATFORM" in
        jetson-*) profile="jetson" ;;
        *)        profile="desktop" ;;
    esac
    info "Starting with Docker (profile: $profile)..."
    docker compose --profile "$profile" up --build
    exit 0
}

run_tests() {
    info "Running test suite..."
    if command -v pytest &>/dev/null; then
        pytest tests/ -v --tb=short
    else
        python3 -m pytest tests/ -v --tb=short
    fi
    exit $?
}

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

main() {
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║     🧠 Jetson-Dream Quick Start         ║"
    echo "╚══════════════════════════════════════════╝"
    echo ""

    # Parse flags
    local USE_DOCKER=false
    local RUN_TEST=false
    local EXTRA_ARGS=()

    for arg in "$@"; do
        case "$arg" in
            --docker)  USE_DOCKER=true ;;
            --test)    RUN_TEST=true ;;
            *)         EXTRA_ARGS+=("$arg") ;;
        esac
    done

    if $RUN_TEST; then
        run_tests
    fi

    if $USE_DOCKER; then
        run_docker
    fi

    # Native mode
    info "Platform: $PLATFORM"
    check_python
    check_torch || install_deps
    check_opencv || install_deps
    check_camera || true
    ensure_models

    echo ""
    info "Launching jetson-dream..."

    # Build optimal args based on platform
    local ARGS=()

    case "$PLATFORM" in
        jetson-nano)
            ARGS+=(--nano --stream)
            info "Jetson Nano detected → --nano mode (160×96, frame skip, FP16)"
            ;;
        jetson-orin)
            ARGS+=(--turbo --stream --process-res low)
            info "Jetson Orin detected → --turbo mode (low resolution)"
            ;;
        desktop-gpu)
            ARGS+=(--turbo --stream --process-res medium)
            info "Desktop GPU detected → --turbo mode (medium resolution)"
            ;;
        cpu-only)
            ARGS+=(--stream --process-res low --frame-skip 2)
            warn "CPU only → low resolution with frame skip"
            ;;
    esac

    # Append user overrides
    ARGS+=("${EXTRA_ARGS[@]}")

    echo ""
    info "Command: python3 -m scripts.main ${ARGS[*]}"
    echo ""
    exec python3 -m scripts.main "${ARGS[@]}"
}

main "$@"
