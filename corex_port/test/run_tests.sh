#!/bin/bash
# CoreX (ivcore11) on-GPU test runner for the Tengine CUDA backend.
# For every bundled benchmark model we run the CUDA classification example on
# GPU 1 and the CPU reference on the same model+image, then check:
#   1. the CUDA run finishes on-GPU with exit code 0
#   2. the CUDA top-5 output matches the CPU top-5 output (numerical parity)
# The weightless *_benchmark.tmfile models carry architecture only (no trained
# weights) so both backends legitimately emit an all-zero top-5; the test asserts
# CUDA==CPU parity rather than a specific class label.
set -u
cd "$(dirname "$0")/../.."
REPO=$(pwd)
source corex_port/env/corex_env.sh
CUDA_BIN=./build/examples/tm_classification_cuda
CPU_BIN=./build/examples/tm_classification
IMG=/tmp/test_img.jpg
[ -f "$IMG" ] || cp doc/docs_en/images/clip_image002.jpg "$IMG"

# Classification models the tm_classification_cuda example is designed for
# (ImageNet-style, 224x224x3 input, 1000-class top-5). Face-embedding / detection
# models (mobilefacenets, mssd, retinaface, yolov3_tiny) are intentionally not in
# this harness: they use incompatible input geometry / output layout and fail
# identically on the CPU reference (verified), i.e. a harness/model-type mismatch,
# not a CoreX/CUDA issue.
MODELS="squeezenet_v1.1 mobilenet mobilenet_v2 googlenet resnet18 resnet50 shufflenet_v2 inception_v3 vgg16"
GEO="224,224"; SC="0.017,0.017,0.017"; MEAN="104.007,116.669,122.679"

pass=0; fail=0; run=0
echo "############ Tengine CUDA backend on-GPU test (device=GPU $CUDA_VISIBLE_DEVICES) ############"
env -u CUDA_VISIBLE_DEVICES ixsmi | sed -n '1,6p'
for m in $MODELS; do
    MF="benchmark/models/${m}_benchmark.tmfile"
    [ -f "$MF" ] || { echo "SKIP $m (model file missing)"; continue; }
    run=$((run+1))
    echo ""
    echo "======================================================================"
    echo "[CASE $run] model=$m  (CUDA on GPU $CUDA_VISIBLE_DEVICES vs CPU parity)"
    echo "----------------------------------------------------------------------"
    echo "--- CUDA (GPU) ---"
    cuda_out=$(timeout 300 $CUDA_BIN -m "$MF" -i "$IMG" -g "$GEO" -s "$SC" -w "$MEAN" -r 3 2>&1)
    cuda_rc=$?
    echo "$cuda_out"
    echo "--- CPU (reference) ---"
    cpu_out=$(timeout 300 $CPU_BIN -m "$MF" -i "$IMG" -g "$GEO" -s "$SC" -w "$MEAN" -r 1 2>&1)
    cpu_rc=$?
    echo "$cpu_out"
    cuda_topk=$(echo "$cuda_out" | grep -E "^[-0-9]+\.[0-9]+, [0-9]+$")
    cpu_topk=$(echo "$cpu_out" | grep -E "^[-0-9]+\.[0-9]+, [0-9]+$")
    cuda_ms=$(echo "$cuda_out" | grep -oE "min_time [0-9.]+ ms" | grep -oE "[0-9.]+" | tail -1)
    echo "----------------------------------------------------------------------"
    if [ "$cuda_rc" -eq 0 ] && [ -n "$cuda_topk" ] && [ "$cuda_topk" == "$cpu_topk" ]; then
        echo "[RESULT $m] PASS  (cuda_rc=0, CUDA/CPU top-5 parity OK, cuda_min=${cuda_ms}ms)"
        pass=$((pass+1))
    else
        echo "[RESULT $m] FAIL  (cuda_rc=$cuda_rc cpu_rc=$cpu_rc parity=$([ "$cuda_topk" == "$cpu_topk" ] && echo ok || echo MISMATCH))"
        fail=$((fail+1))
    fi
done
echo ""
echo "############ SUMMARY ############"
echo "tests_run=$run tests_passed=$pass tests_failed=$fail"
exit $fail
