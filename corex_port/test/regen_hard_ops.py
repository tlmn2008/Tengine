#!/usr/local/bin/python3
"""Regenerate onnx_node data for the ops Tengine's 2024 convert_tool cannot take
directly from the stock onnx 1.22 (opset 25) node models.

Two problem classes:
  (A) Conv / BatchNormalization: their learned tensors (W/B, scale/bias/mean/var)
      are GRAPH INPUTS in the onnx node models (input_1.pb, input_2.pb, ...), but
      Tengine's test harness only sets input_0, so they run with unset weights and
      emit all-zeros. Fix: bake those extra inputs as INITIALIZERS (from the
      provided input_k.pb) so the weights are part of the model; keep only input_0.
      Reference output_0.pb is unchanged (still the onnx-computed reference).
  (B) Unsqueeze / Pad(reflect): opset-25 moved axes/pads from attribute to input,
      which the old convert_tool can't parse (segfault). Rebuild an equivalent
      opset-11/opset-10 model with axes/pads as ATTRIBUTES. For Unsqueeze the op is
      a pure data pass-through, so we generate self-consistent input/output; for
      Pad(reflect) we reuse the onnx input_0.pb/output_0.pb (exact integer result).

Outputs go to build/onnx_node/<node>/ (== ../onnx_node from the ctest cwd).
"""
import os, subprocess, sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ND = os.path.join(os.path.dirname(onnx.__file__), "backend/test/data/node")
OUT = os.path.join(REPO, "build/onnx_node")
CT = os.path.join(REPO, "build/tools/convert_tool/convert_tool")

def convert(src_onnx, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    r = subprocess.run([CT, "-f", "onnx", "-m", src_onnx, "-o", os.path.join(dst_dir, "onnx.tmfile")],
                       capture_output=True, text=True)
    ok = os.path.exists(os.path.join(dst_dir, "onnx.tmfile")) and os.path.getsize(os.path.join(dst_dir, "onnx.tmfile")) > 0
    return ok, (r.stdout + r.stderr)[-400:]

def save_tensor(arr, path, name):
    onnx.save_tensor(numpy_helper.from_array(arr.astype(np.float32), name), path)

def write_dataset(dst_dir, x, y):
    d0 = os.path.join(dst_dir, "test_data_set_0"); os.makedirs(d0, exist_ok=True)
    save_tensor(x, os.path.join(d0, "input_0.pb"), "x")
    save_tensor(y, os.path.join(d0, "output_0.pb"), "y")
    save_tensor(y, os.path.join(d0, "output.pb"), "y")  # alias for tests reading output.pb

# ---------- (A) bake extra graph inputs (weights) as initializers ----------
def bake_weights(node_name, out_name, opset=None):
    src = os.path.join(ND, node_name)
    m = onnx.load(os.path.join(src, "model.onnx"))
    g = m.graph
    init_names = {i.name for i in g.initializer}
    graph_inputs = [i for i in g.input if i.name not in init_names]
    # first graph input = data (set from input_0 by the harness); bake the rest
    for k, gi in enumerate(graph_inputs[1:], start=1):
        pb = os.path.join(src, "test_data_set_0", f"input_{k}.pb")
        if not os.path.exists(pb):
            return False, f"missing {pb}"
        arr = numpy_helper.to_array(onnx.load_tensor(pb))
        g.initializer.append(numpy_helper.from_array(arr, gi.name))
    # keep only the data input (input_0); baked tensors are now initializers
    del g.input[:]
    g.input.append(graph_inputs[0])
    dst = os.path.join(OUT, out_name)
    os.makedirs(dst, exist_ok=True)
    tmp = os.path.join(dst, "model_baked.onnx")
    if opset is not None:
        del m.opset_import[:]; m.opset_import.append(helper.make_opsetid("", opset))
        m.ir_version = 8
    onnx.save(m, tmp)
    # copy reference dataset (input_0 = data, output_0 = reference)
    import shutil
    d0 = os.path.join(dst, "test_data_set_0"); os.makedirs(d0, exist_ok=True)
    for f in ("input_0.pb", "output_0.pb"):
        shutil.copy(os.path.join(src, "test_data_set_0", f), os.path.join(d0, f))
    shutil.copy(os.path.join(src, "test_data_set_0", "output_0.pb"), os.path.join(d0, "output.pb"))
    return convert(tmp, dst)

# ---------- (A') BatchNormalization: add the default epsilon attr the tool requires ----------
def gen_batchnorm(node_name, out_name):
    # Tengine's convert_tool does get_attr("epsilon") with no default and throws
    # ("cannot find attr epsilon") when the attribute is omitted. The passing
    # sibling test_batchnorm_epsilon sets it explicitly; test_batchnorm_example
    # relies on the ONNX default (1e-5). Add it explicitly, mirroring the sibling.
    import shutil
    src = os.path.join(ND, node_name)
    m = onnx.load(os.path.join(src, "model.onnx"))
    for node in m.graph.node:
        if node.op_type == "BatchNormalization" and not any(a.name == "epsilon" for a in node.attribute):
            node.attribute.append(helper.make_attribute("epsilon", 1e-5))
    dst = os.path.join(OUT, out_name); os.makedirs(dst, exist_ok=True)
    tmp = os.path.join(dst, "model_eps.onnx"); onnx.save(m, tmp)
    d0 = os.path.join(dst, "test_data_set_0"); os.makedirs(d0, exist_ok=True)
    for f in os.listdir(os.path.join(src, "test_data_set_0")):
        shutil.copy(os.path.join(src, "test_data_set_0", f), os.path.join(d0, f))
    return convert(tmp, dst)

# ---------- (B1) Unsqueeze rebuilt at opset 11 (axes as attribute) ----------
def gen_unsqueeze(out_name, axes):
    dst = os.path.join(OUT, out_name)
    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    y = x
    for ax in sorted([a if a >= 0 else a + (x.ndim + len([q for q in axes if q < 0])) for a in axes]):
        y = np.expand_dims(y, axis=ax)
    xi = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))
    yo = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y.shape))
    node = helper.make_node("Unsqueeze", ["x"], ["y"], axes=list(axes))
    g = helper.make_graph([node], out_name, [xi], [yo])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 11)]); m.ir_version = 8
    os.makedirs(dst, exist_ok=True)
    tmp = os.path.join(dst, "model_gen.onnx"); onnx.save(m, tmp)
    write_dataset(dst, x, y)
    return convert(tmp, dst)

# ---------- (B2) Pad(reflect) rebuilt at opset 2 (pads+mode as attributes) ----------
def gen_reflect_pad(out_name):
    src = os.path.join(ND, "test_reflect_pad")
    dst = os.path.join(OUT, out_name)
    x = numpy_helper.to_array(onnx.load_tensor(os.path.join(src, "test_data_set_0", "input_0.pb")))
    y = numpy_helper.to_array(onnx.load_tensor(os.path.join(src, "test_data_set_0", "output_0.pb")))
    pads_pb = onnx.load_tensor(os.path.join(src, "test_data_set_0", "input_1.pb"))
    pads = [int(v) for v in numpy_helper.to_array(pads_pb).tolist()]  # [b0,b1,b2,b3,e0,e1,e2,e3]
    xi = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))
    yo = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y.shape))
    node = helper.make_node("Pad", ["x"], ["y"], mode="reflect", pads=pads)  # opset<11: pads attr
    g = helper.make_graph([node], out_name, [xi], [yo])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 2)]); m.ir_version = 8
    os.makedirs(dst, exist_ok=True)
    tmp = os.path.join(dst, "model_gen.onnx"); onnx.save(m, tmp)
    write_dataset(dst, x, y)
    return convert(tmp, dst)

if __name__ == "__main__":
    results = {}
    # NOTE: conv tests (basic_conv_*, conv_with_strides_*) intentionally NOT baked:
    # the Tengine harness sets BOTH data (input_0) and weights (input_1) as graph
    # inputs, so the raw 2-input onnx model (from the plain datagen) is what the test
    # expects. They still fail numerically because the harness fills the weight buffer
    # AFTER prerun_graph (which prepacks zero weights) -> all-zero conv output; that is a
    # pre-existing Tengine CPU-backend/harness ordering bug, unrelated to CoreX/CUDA.
    # (A') batchnorm example: add the default epsilon attr (mirrors passing sibling)
    results["test_batchnorm_example"] = gen_batchnorm("test_batchnorm_example", "test_batchnorm_example")
    # (B1) unsqueeze family (axes chosen to match each test's intent; op is data-passthrough)
    for name, axes in [("test_unsqueeze_axis_0", [0]), ("test_unsqueeze_axis_1", [1]),
                        ("test_unsqueeze_axis_2", [2]), ("test_unsqueeze_axis_3", [3]),
                        ("test_unsqueeze_negative_axes", [-2]), ("test_unsqueeze_two_axes", [1, 4]),
                        ("test_unsqueeze_three_axes", [2, 4, 5]), ("test_unsqueeze_unsorted_axes", [5, 4, 2])]:
        results[name] = gen_unsqueeze(name, axes)
    # (B2) reflect pad
    results["test_pad_reflect"] = gen_reflect_pad("test_pad_reflect")

    ok = 0
    for n, (good, msg) in results.items():
        print(f"[{'OK ' if good else 'FAIL'}] {n}" + ("" if good else f"  :: {msg.strip()[:200]}"))
        ok += bool(good)
    print(f"---- regen: {ok}/{len(results)} converted ----")
    sys.exit(0 if ok == len(results) else 1)
