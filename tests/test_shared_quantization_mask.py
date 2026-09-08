from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import onnx
from onnx import TensorProto, helper
import pytest

from pilot_registry import canonical_hash as pilot_canonical_hash
from topic_c.shared_quantization_mask import (
    canonical_hash,
    compute_attachment_contract,
    enforce_frozen_compute_input_policy,
    normalized_attachment_topology,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "src" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_complete(path: Path, document: dict, *, marker: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".complete").write_text((marker or sha256_file(path)) + "\n")


def source_model(path: Path) -> None:
    initializers = [
        helper.make_tensor("weight.common", TensorProto.FLOAT, [2, 1, 1, 1], [1.0, 2.0]),
        helper.make_tensor("weight.extra", TensorProto.FLOAT, [2, 2, 1, 1], [3.0, 4.0, 5.0, 6.0]),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "weight.common"], ["common"], name="/model.1/conv/Conv"),
        # This excluded sibling consumes a tensor that is quantized for another
        # consumer.  Explicit ORT node selection can incorrectly route this
        # raw baseline edge through the shared Q/DQ pair.
        helper.make_node("Conv", ["common", "weight.extra"], ["extra"], name="/model.10/m.0/conv/Conv"),
        helper.make_node("Add", ["common", "common"], ["output"], name="/model.10/Add"),
        # ModelOpt is allowed to remove/fold non-compute normalization nodes.
        helper.make_node("Cast", ["input"], ["unused.cast"], name="/model.28/Cast_2", to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes,
        "source",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("extra", TensorProto.FLOAT, [1, 2, 2, 2]),
        ],
        initializers,
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), path)


def quantized_model(path: Path, *, include_extra: bool) -> None:
    initializers = [
        helper.make_tensor("weight.common", TensorProto.FLOAT, [2, 1, 1, 1], [1.0, 2.0]),
        helper.make_tensor("weight.extra", TensorProto.FLOAT, [2, 2, 1, 1], [3.0, 4.0, 5.0, 6.0]),
        helper.make_tensor("a.scale", TensorProto.FLOAT, [], [0.1]),
        helper.make_tensor("a.zero", TensorProto.INT8, [], [0]),
        helper.make_tensor("w.scale", TensorProto.FLOAT, [2], [0.1, 0.2]),
        helper.make_tensor("w.zero", TensorProto.INT8, [2], [0, 0]),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "a.scale", "a.zero"], ["input.q"], name="input.Q"),
        helper.make_node("DequantizeLinear", ["input.q", "a.scale", "a.zero"], ["input.dq"], name="input.DQ"),
        helper.make_node("QuantizeLinear", ["weight.common", "w.scale", "w.zero"], ["wc.q"], name="wc.Q", axis=0),
        helper.make_node("DequantizeLinear", ["wc.q", "w.scale", "w.zero"], ["wc.dq"], name="wc.DQ", axis=0),
        helper.make_node("Conv", ["input.dq", "wc.dq"], ["common"], name="/model.1/conv/Conv"),
        helper.make_node("QuantizeLinear", ["common", "a.scale", "a.zero"], ["common.q"], name="common.Q"),
        helper.make_node("DequantizeLinear", ["common.q", "a.scale", "a.zero"], ["common.dq"], name="common.DQ"),
        # Baseline FP8 attaches Q/DQ to only one residual input.  The other
        # source-produced edge must be replayed through no_quantize_inputs.
        helper.make_node("Add", ["common.dq", "common"], ["output"], name="/model.10/Add"),
    ]
    if include_extra:
        nodes.extend(
            [
                helper.make_node("QuantizeLinear", ["weight.extra", "w.scale", "w.zero"], ["we.q"], name="we.Q", axis=0),
                helper.make_node("DequantizeLinear", ["we.q", "w.scale", "w.zero"], ["we.dq"], name="we.DQ", axis=0),
                helper.make_node("Conv", ["common.dq", "we.dq"], ["extra"], name="/model.10/m.0/conv/Conv"),
            ]
        )
    else:
        nodes.append(
            helper.make_node(
                "Conv", ["common", "weight.extra"], ["extra"],
                name="/model.10/m.0/conv/Conv",
            )
        )
    graph = helper.make_graph(
        nodes,
        "quantized",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("extra", TensorProto.FLOAT, [1, 2, 2, 2]),
        ],
        initializers,
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), path)


def frozen_fixture(tmp_path: Path) -> dict:
    source = tmp_path / "source.onnx"
    baseline_fp8 = tmp_path / "baseline.fp8.onnx"
    baseline_int8 = tmp_path / "baseline.int8.onnx"
    source_model(source)
    quantized_model(baseline_fp8, include_extra=False)
    quantized_model(baseline_int8, include_extra=True)

    source_registry = tmp_path / "source.json"
    int8_registry = tmp_path / "int8.json"
    fp8_registry = tmp_path / "fp8.json"
    source_document = {
        "dataset": "voc",
        "model": "toy",
        "imgsz": 16,
        "onnx": str(source),
        "onnx_sha256": sha256_file(source),
    }
    write_complete(source_registry, source_document)
    calibration = tmp_path / "calibration.json"
    calibration_document = {"schema_version": 1, "dataset": "voc", "split": "train", "records": []}
    calibration_document["calibration_sha256"] = pilot_canonical_hash(
        calibration_document, "calibration_sha256"
    )
    write_complete(calibration, calibration_document, marker=calibration_document["calibration_sha256"])
    common_registry = {
        "dataset": "voc",
        "model": "toy",
        "imgsz": 16,
        "source_onnx_sha256": sha256_file(source),
        "calibration_sha256": calibration_document["calibration_sha256"],
        "calibration_method": "entropy",
    }
    write_complete(
        int8_registry,
        {
            **common_registry,
            "precision": "int8-entropy",
            "output_onnx": str(baseline_int8),
            "output_onnx_sha256": sha256_file(baseline_int8),
        },
    )
    write_complete(
        fp8_registry,
        {
            **common_registry,
            "precision": "fp8",
            "output_onnx": str(baseline_fp8),
            "output_onnx_sha256": sha256_file(baseline_fp8),
        },
    )
    return {
        "source": source,
        "source_document": source_document,
        "source_registry": source_registry,
        "int8_registry": int8_registry,
        "fp8_registry": fp8_registry,
        "baseline_fp8": baseline_fp8,
        "baseline_int8": baseline_int8,
        "calibration": calibration,
        "calibration_sha256": calibration_document["calibration_sha256"],
    }


def test_freeze_recovers_fp8_sites_and_records_int8_only_site(tmp_path: Path) -> None:
    module = load_script("freeze_shared_quantization_mask.py")
    fixture = frozen_fixture(tmp_path)
    output = tmp_path / "mask.json"
    block = {
        "id": "toy",
        "dataset": "voc",
        "model": "toy",
        "imgsz": 16,
        "source_onnx_registry": str(fixture["source_registry"]),
        "baseline_int8_registry": str(fixture["int8_registry"]),
        "baseline_fp8_registry": str(fixture["fp8_registry"]),
        "calibration_list": str(fixture["calibration"]),
    }

    mask = module.freeze_block(tmp_path, "shared_mask_pilot_v1", block, output)

    assert [item["name"] for item in mask["nodes"]] == ["/model.1/conv/Conv", "/model.10/Add"]
    assert mask["node_type_counts"] == {"Add": 1, "Conv": 1}
    assert mask["schema_version"] == 3
    assert mask["policy"] == "frozen_baseline_fp8_compute_attachment_contract_v3"
    assert mask["edge_policy"]["no_quantize_inputs"] == [
        {
            "source_node": "/model.1/conv/Conv",
            "source_op": "Conv",
            "source_output_index": 0,
            "target_node": "/model.10/Add",
            "target_op": "Add",
            "target_input_index": 1,
            "input_name": "common",
        },
        {
            "source_node": "/model.1/conv/Conv",
            "source_op": "Conv",
            "source_output_index": 0,
            "target_node": "/model.10/m.0/conv/Conv",
            "target_op": "Conv",
            "target_input_index": 0,
            "input_name": "common",
        },
    ]
    policy = mask["compute_input_policy"]
    assert policy["attached_inputs"] == 3
    assert policy["raw_inputs"] == 3
    assert mask["baseline_int8"]["extra_compute_sites"] == [
        {"name": "/model.10/m.0/conv/Conv", "op_type": "Conv"}
    ]
    assert output.with_suffix(".json.complete").read_text().strip() == mask["mask_sha256"]


def test_real_pilot_config_is_self_hashed_and_has_three_declared_blocks() -> None:
    config = json.loads((ROOT / "configs" / "shared_mask_pilot_v1.json").read_text())

    assert config["config_sha256"] == canonical_hash(config, "config_sha256")
    assert [block["id"] for block in config["blocks"]] == [
        "voc_yolo11m",
        "voc_rtdetr_l",
        "kitti_retinanet_r50_fpn_v2",
    ]

    module = load_script("freeze_shared_quantization_mask.py")
    v2 = {**config, "attempt": "shared_mask_pilot_v2"}
    v2["config_sha256"] = canonical_hash(v2, "config_sha256")
    module._validate_config(v2)


def test_quantizer_resolves_literal_mask_names_as_exact_regexes(tmp_path: Path) -> None:
    freeze_module = load_script("freeze_shared_quantization_mask.py")
    quantize_module = load_script("quantize_yolo_onnx.py")
    fixture = frozen_fixture(tmp_path)
    output = tmp_path / "mask.json"
    block = {
        "id": "toy",
        "dataset": "voc",
        "model": "toy",
        "imgsz": 16,
        "source_onnx_registry": str(fixture["source_registry"]),
        "baseline_int8_registry": str(fixture["int8_registry"]),
        "baseline_fp8_registry": str(fixture["fp8_registry"]),
        "calibration_list": str(fixture["calibration"]),
    }
    mask = freeze_module.freeze_block(tmp_path, "shared_mask_pilot_v1", block, output)
    graph = onnx.load(fixture["source"]).graph

    resolved, op_types, patterns = quantize_module.resolve_node_mask(
        output,
        source=fixture["source_document"],
        source_registry_file_sha256=sha256_file(fixture["source_registry"]),
        source_onnx_sha256=sha256_file(fixture["source"]),
        graph=graph,
        mode="int8-entropy",
        imgsz=16,
        calibration_sha256=fixture["calibration_sha256"],
        requested_op_types=None,
    )

    assert resolved["mask_sha256"] == mask["mask_sha256"]
    assert op_types == ["Add", "Conv"]
    assert patterns == [r"^/model\.1/conv/Conv$", r"^/model\.10/Add$"]


def test_quantizer_renders_complete_raw_edge_policy_without_name_heuristics(
    tmp_path: Path, monkeypatch
) -> None:
    freeze_module = load_script("freeze_shared_quantization_mask.py")
    quantize_module = load_script("quantize_yolo_onnx.py")
    fixture = frozen_fixture(tmp_path)
    output = tmp_path / "mask.json"
    block = {
        "id": "toy", "dataset": "voc", "model": "toy", "imgsz": 16,
        "source_onnx_registry": str(fixture["source_registry"]),
        "baseline_int8_registry": str(fixture["int8_registry"]),
        "baseline_fp8_registry": str(fixture["fp8_registry"]),
        "calibration_list": str(fixture["calibration"]),
    }
    mask = freeze_module.freeze_block(tmp_path, "shared_mask_pilot_v1", block, output)

    class FakeNode:
        def __init__(self, *, op, name):
            self.op, self.name = op, name

    monkeypatch.setitem(sys.modules, "onnx_graphsurgeon", types.SimpleNamespace(Node=FakeNode))
    tuples = quantize_module.materialize_no_quantize_inputs(
        mask, onnx.load(fixture["source"]).graph
    )
    assert len(tuples) == 2
    assert {
        (source.name, source.op, target.name, target.op, tensor)
        for source, target, tensor in tuples
    } == {
        ("/model.1/conv/Conv", "Conv", "/model.10/Add", "Add", "common"),
        (
            "/model.1/conv/Conv", "Conv", "/model.10/m.0/conv/Conv", "Conv", "common"
        ),
    }


def test_target_slot_enforcement_removes_surplus_activation_and_weight_qdq(
    tmp_path: Path,
) -> None:
    freeze_module = load_script("freeze_shared_quantization_mask.py")
    fixture = frozen_fixture(tmp_path)
    mask_path = tmp_path / "mask.json"
    mask = freeze_module.freeze_block(
        tmp_path,
        "shared_mask_pilot_v2",
        {
            "id": "toy", "dataset": "voc", "model": "toy", "imgsz": 16,
            "source_onnx_registry": str(fixture["source_registry"]),
            "baseline_int8_registry": str(fixture["int8_registry"]),
            "baseline_fp8_registry": str(fixture["fp8_registry"]),
            "calibration_list": str(fixture["calibration"]),
        },
        mask_path,
    )
    output = tmp_path / "enforced.int8.onnx"

    audit = enforce_frozen_compute_input_policy(
        fixture["source"], fixture["baseline_int8"], mask, output
    )

    assert audit["bypassed_inputs_count"] == 2
    assert {
        (item["node_name"], item["input_index"], item["anchor_kind"])
        for item in audit["bypassed_inputs"]
    } == {
        ("/model.10/m.0/conv/Conv", 0, "node_output"),
        ("/model.10/m.0/conv/Conv", 1, "initializer"),
    }
    baseline = normalized_attachment_topology(fixture["source"], fixture["baseline_fp8"])
    realized = normalized_attachment_topology(fixture["source"], output)
    assert compute_attachment_contract(realized) == compute_attachment_contract(baseline)


def test_target_slot_enforcement_refuses_a_missing_required_attachment(tmp_path: Path) -> None:
    freeze_module = load_script("freeze_shared_quantization_mask.py")
    fixture = frozen_fixture(tmp_path)
    mask_path = tmp_path / "mask.json"
    mask = freeze_module.freeze_block(
        tmp_path,
        "shared_mask_pilot_v2",
        {
            "id": "toy", "dataset": "voc", "model": "toy", "imgsz": 16,
            "source_onnx_registry": str(fixture["source_registry"]),
            "baseline_int8_registry": str(fixture["int8_registry"]),
            "baseline_fp8_registry": str(fixture["fp8_registry"]),
            "calibration_list": str(fixture["calibration"]),
        },
        mask_path,
    )
    missing = tmp_path / "missing-required.onnx"
    model = onnx.load(fixture["baseline_int8"])
    common = next(node for node in model.graph.node if node.name == "/model.1/conv/Conv")
    common.input[1] = "weight.common"
    onnx.save(model, missing)

    with pytest.raises(ValueError, match="missing a required baseline FP8 Q/DQ input"):
        enforce_frozen_compute_input_policy(
            fixture["source"], missing, mask, tmp_path / "must-not-exist.onnx"
        )
    assert not (tmp_path / "must-not-exist.onnx").exists()


def test_verifier_accepts_equal_topology_and_rejects_extra_int8_site(tmp_path: Path) -> None:
    freeze_module = load_script("freeze_shared_quantization_mask.py")
    verify_module = load_script("verify_shared_quantization_mask.py")
    fixture = frozen_fixture(tmp_path)
    mask_path = tmp_path / "mask.json"
    block = {
        "id": "toy",
        "dataset": "voc",
        "model": "toy",
        "imgsz": 16,
        "source_onnx_registry": str(fixture["source_registry"]),
        "baseline_int8_registry": str(fixture["int8_registry"]),
        "baseline_fp8_registry": str(fixture["fp8_registry"]),
        "calibration_list": str(fixture["calibration"]),
    }
    mask = freeze_module.freeze_block(tmp_path, "shared_mask_pilot_v1", block, mask_path)

    enforced_int8 = tmp_path / "enforced.int8.onnx"
    enforcement = enforce_frozen_compute_input_policy(
        fixture["source"], fixture["baseline_int8"], mask, enforced_int8
    )

    def shared_registry(path: Path, precision: str, onnx_path: Path) -> None:
        is_fp8 = precision == "fp8"
        write_complete(
            path,
            {
                "dataset": "voc",
                "model": "toy",
                "imgsz": 16,
                "precision": precision,
                "source_onnx_sha256": sha256_file(fixture["source"]),
                "calibration_sha256": fixture["calibration_sha256"],
                "output_onnx": str(onnx_path),
                "output_onnx_sha256": sha256_file(onnx_path),
                "node_mask": str(mask_path),
                "node_mask_file_sha256": sha256_file(mask_path),
                "node_mask_sha256": mask["mask_sha256"],
                "nodes_to_quantize_count": len(mask["nodes"]),
                "no_quantize_inputs_count": len(mask["edge_policy"]["no_quantize_inputs"]),
                "edge_policy_sha256": mask["edge_policy"]["edge_policy_sha256"],
                "compute_input_policy_sha256": mask["compute_input_policy"]["policy_sha256"],
                "baseline_fp8_onnx_sha256": mask["baseline_fp8"]["onnx_sha256"],
                "post_enforcement_compute_attachment_sha256": mask[
                    "baseline_fp8_normalized_topology"
                ]["compute_attachment_sha256"],
                "fp8_baseline_byte_replay": is_fp8,
                "shared_mask_role": (
                    "frozen_baseline_fp8_control"
                    if is_fp8
                    else "int8_aligned_to_frozen_fp8_compute_inputs"
                ),
                "bypassed_compute_inputs_count": (
                    0 if is_fp8 else enforcement["bypassed_inputs_count"]
                ),
            },
        )

    shared_int8 = tmp_path / "shared.int8.json"
    shared_fp8 = tmp_path / "shared.fp8.json"
    shared_registry(shared_int8, "int8-entropy", enforced_int8)
    shared_registry(shared_fp8, "fp8", fixture["baseline_fp8"])

    report = verify_module.verify(mask_path, shared_int8, shared_fp8, tmp_path / "pass.json")
    assert report["status"] == "pass"
    assert report["mask"]["nodes"] == 2

    bad_int8 = tmp_path / "bad.int8.json"
    shared_registry(bad_int8, "int8-entropy", fixture["baseline_int8"])
    with pytest.raises(ValueError, match="attachment topologies differ"):
        verify_module.verify(mask_path, bad_int8, shared_fp8, tmp_path / "refused.json")

    # Equal shared outputs are still invalid if they do not replay baseline FP8.
    nonreplay_int8 = tmp_path / "nonreplay.int8.json"
    nonreplay_fp8 = tmp_path / "nonreplay.fp8.json"
    shared_registry(nonreplay_int8, "int8-entropy", fixture["baseline_int8"])
    shared_registry(nonreplay_fp8, "fp8", fixture["baseline_int8"])
    with pytest.raises(ValueError, match="byte-identical to the frozen baseline FP8"):
        verify_module.verify(mask_path, nonreplay_int8, nonreplay_fp8, tmp_path / "nonreplay.json")


def test_topology_tolerates_removed_noncompute_but_refuses_missing_masked_compute(tmp_path: Path) -> None:
    from topic_c.shared_quantization_mask import normalized_attachment_topology

    fixture = frozen_fixture(tmp_path)
    # The source-only /model.28/Cast_2 is absent from this quantized graph.
    topology = normalized_attachment_topology(
        fixture["source"],
        fixture["baseline_fp8"],
        required_node_names={"/model.1/conv/Conv", "/model.10/Add"},
    )
    assert topology["quantize_linear_nodes"] == 3

    missing = tmp_path / "missing-masked-compute.onnx"
    model = onnx.load(fixture["baseline_fp8"])
    next(node for node in model.graph.node if node.name == "/model.1/conv/Conv").name = "/folded/Conv"
    onnx.save(model, missing)
    with pytest.raises(ValueError, match="required compute node is absent or changed"):
        normalized_attachment_topology(
            fixture["source"],
            missing,
            required_node_names={"/model.1/conv/Conv"},
        )
