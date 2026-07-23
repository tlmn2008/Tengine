#!/bin/bash
# Generate the ../onnx_node/<op>/ test data that Tengine's 75 test_onnx_op_* ctests
# expect, from the installed onnx package's ONNX backend node conformance data
# (onnx 1.22.0: .../site-packages/onnx/backend/test/data/node/<node>/{model.onnx,
# test_data_set_*/*.pb}). For each op: copy the test_data_set_* dirs and convert
# model.onnx -> onnx.tmfile with Tengine's own convert_tool (onnx2tengine).
# Output dir = build/onnx_node/<node>  (== ../onnx_node from the ctest cwd build/tests).
set -u
cd "$(dirname "$0")/../.."          # repo root
REPO=$(pwd)
source corex_port/env/corex_env.sh
CT="$REPO/build/tools/convert_tool/convert_tool"
ND=$(/usr/local/bin/python3 -c "import onnx,os;print(os.path.join(os.path.dirname(onnx.__file__),'backend/test/data/node'))")
OUT="$REPO/build/onnx_node"
MAP="$REPO/corex_port/test/test_node_map.txt"   # lines: test_target|node
rm -rf "$OUT"; mkdir -p "$OUT"
echo "onnx node data source: $ND"
echo "convert_tool: $CT"
gen_ok=0; gen_fail=0; synth=0
while IFS='|' read -r target node; do
    [ -z "$node" ] && continue
    src="$node"
    # Tengine node names that differ from the stock onnx node dir name:
    case "$node" in
        test_pad_reflect) src="test_reflect_pad" ;;   # same op, different name
    esac
    dst="$OUT/$node"; mkdir -p "$dst"
    if [ -d "$ND/$src" ]; then
        # copy every test_data_set_* (covers multi-input ops: input_0/1/2..., output_0)
        cp -r "$ND/$src"/test_data_set_* "$dst"/ 2>/dev/null
        # some Tengine tests read output.pb instead of output_0.pb -> provide alias
        for ds in "$dst"/test_data_set_*; do
            [ -f "$ds/output_0.pb" ] && [ ! -f "$ds/output.pb" ] && cp "$ds/output_0.pb" "$ds/output.pb"
        done
        if "$CT" -f onnx -m "$ND/$src/model.onnx" -o "$dst/onnx.tmfile" >/dev/null 2>&1 && [ -s "$dst/onnx.tmfile" ]; then
            echo "[GEN OK]   $node  (src onnx node: $src)"
            gen_ok=$((gen_ok+1))
        else
            echo "[GEN FAIL] $node  (convert_tool could not convert $src/model.onnx)"
            gen_fail=$((gen_fail+1))
        fi
    else
        echo "[NO ONNX NODE] $node  (no matching dir under onnx backend node data; will try synth separately)"
        synth=$((synth+1))
    fi
done < "$MAP"
echo "---- datagen summary: GEN_OK=$gen_ok GEN_FAIL=$gen_fail NO_ONNX_NODE=$synth ----"
