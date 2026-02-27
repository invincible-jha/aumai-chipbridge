"""Comprehensive tests for aumai-chipbridge core module.

Covers: GraphAnalyzer, HardwareMapper, OptimizationEngine, PerformancePredictor,
CrossHardwareComparator, HardwareRegistry, InferenceOptimizer, CrossCompiler
and all Pydantic models.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from aumai_chipbridge.core import (
    HARDWARE_PROFILES,
    CrossCompiler,
    CrossHardwareComparator,
    GraphAnalyzer,
    HardwareMapper,
    HardwareRegistry,
    InferenceOptimizer,
    OptimizationEngine,
    PerformancePredictor,
)
from aumai_chipbridge.models import (
    GraphOperator,
    HardwareProfile,
    HardwareTarget,
    InferencePrediction,
    ModelGraph,
    ModelProfile,
    OperatorPrediction,
    OperatorType,
    OptimizationPass,
    OptimizationPlan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def matmul_op() -> GraphOperator:
    return GraphOperator(
        op_id="mm_0",
        op_type=OperatorType.MATMUL,
        input_shapes=[(1, 768, 768)],
        output_shape=(1, 768),
    )


@pytest.fixture()
def attention_op() -> GraphOperator:
    return GraphOperator(
        op_id="attn_0",
        op_type=OperatorType.ATTENTION,
        input_shapes=[(1, 128, 768)],
        output_shape=(1, 128, 768),
    )


@pytest.fixture()
def simple_graph(matmul_op: GraphOperator, attention_op: GraphOperator) -> ModelGraph:
    layernorm_op = GraphOperator(
        op_id="ln_0",
        op_type=OperatorType.LAYERNORM,
        input_shapes=[(1, 128, 768)],
        output_shape=(1, 128, 768),
    )
    return ModelGraph(
        name="test_model",
        operators=[attention_op, matmul_op, layernorm_op],
        edges=[("attn_0", "mm_0"), ("mm_0", "ln_0")],
    )


@pytest.fixture()
def transformer_graph() -> ModelGraph:
    ops = [
        GraphOperator(op_id="embed", op_type=OperatorType.EMBEDDING,
                      input_shapes=[(1, 128)], output_shape=(1, 128, 768),
                      attributes={"embed_dim": 768}),
        GraphOperator(op_id="attn_0", op_type=OperatorType.ATTENTION,
                      input_shapes=[(1, 128, 768)], output_shape=(1, 128, 768)),
        GraphOperator(op_id="ln_0", op_type=OperatorType.LAYERNORM,
                      input_shapes=[(1, 128, 768)], output_shape=(1, 128, 768)),
        GraphOperator(op_id="mm_0", op_type=OperatorType.MATMUL,
                      input_shapes=[(1, 128, 768)], output_shape=(1, 128, 3072)),
        GraphOperator(op_id="gelu_0", op_type=OperatorType.GELU,
                      input_shapes=[(1, 128, 3072)], output_shape=(1, 128, 3072)),
    ]
    edges = [
        ("embed", "attn_0"),
        ("attn_0", "ln_0"),
        ("ln_0", "mm_0"),
        ("mm_0", "gelu_0"),
    ]
    return ModelGraph(name="transformer", operators=ops, edges=edges)


@pytest.fixture()
def registry() -> HardwareRegistry:
    return HardwareRegistry()


@pytest.fixture()
def analyzer() -> GraphAnalyzer:
    return GraphAnalyzer()


@pytest.fixture()
def mapper() -> HardwareMapper:
    return HardwareMapper()


@pytest.fixture()
def engine() -> OptimizationEngine:
    return OptimizationEngine()


@pytest.fixture()
def predictor() -> PerformancePredictor:
    return PerformancePredictor()


@pytest.fixture()
def comparator() -> CrossHardwareComparator:
    return CrossHardwareComparator()


@pytest.fixture()
def small_model() -> ModelProfile:
    return ModelProfile(
        model_name="bert-base",
        parameter_count=110_000_000,
        flops_per_inference=21_800_000_000.0,
        memory_footprint_mb=440.0,
        has_attention=True,
    )


# ---------------------------------------------------------------------------
# Model tests — HardwareTarget / OperatorType
# ---------------------------------------------------------------------------


class TestEnums:
    def test_hardware_targets_include_a100(self) -> None:
        assert HardwareTarget.A100 in HardwareTarget.__members__.values()

    def test_hardware_targets_include_cpu_x86(self) -> None:
        assert HardwareTarget.CPU_X86 in HardwareTarget.__members__.values()

    def test_operator_types(self) -> None:
        assert OperatorType.MATMUL.value == "matmul"
        assert OperatorType.ATTENTION.value == "attention"


# ---------------------------------------------------------------------------
# Model tests — GraphOperator
# ---------------------------------------------------------------------------


class TestGraphOperatorModel:
    def test_basic_creation(self, matmul_op: GraphOperator) -> None:
        assert matmul_op.op_id == "mm_0"
        assert matmul_op.op_type == OperatorType.MATMUL

    def test_blank_op_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GraphOperator(op_id="   ", op_type=OperatorType.MATMUL)

    def test_op_id_stripped(self) -> None:
        op = GraphOperator(op_id="  my_op  ", op_type=OperatorType.CONV2D)
        assert op.op_id == "my_op"

    def test_default_attributes(self) -> None:
        op = GraphOperator(op_id="op1", op_type=OperatorType.SOFTMAX)
        assert op.attributes == {}

    def test_custom_attributes(self) -> None:
        op = GraphOperator(op_id="op2", op_type=OperatorType.CONV2D,
                           attributes={"stride": 2, "padding": 1})
        assert op.attributes["stride"] == 2


# ---------------------------------------------------------------------------
# Model tests — ModelGraph
# ---------------------------------------------------------------------------


class TestModelGraphModel:
    def test_empty_graph(self) -> None:
        g = ModelGraph(name="empty")
        assert g.operators == []
        assert g.edges == []

    def test_operator_by_id_found(self, simple_graph: ModelGraph) -> None:
        op = simple_graph.operator_by_id("attn_0")
        assert op is not None
        assert op.op_type == OperatorType.ATTENTION

    def test_operator_by_id_not_found(self, simple_graph: ModelGraph) -> None:
        op = simple_graph.operator_by_id("nonexistent")
        assert op is None


# ---------------------------------------------------------------------------
# Model tests — HardwareProfile / ModelProfile / OptimizationPlan
# ---------------------------------------------------------------------------


class TestHardwareProfileModel:
    def test_a100_profile(self) -> None:
        profile = HARDWARE_PROFILES[HardwareTarget.A100]
        assert profile.compute_tflops > 0
        assert profile.has_tensor_cores is True
        assert profile.supports_fp16 is True

    def test_cpu_x86_no_tensor_cores(self) -> None:
        profile = HARDWARE_PROFILES[HardwareTarget.CPU_X86]
        assert profile.has_tensor_cores is False

    def test_invalid_compute_tflops(self) -> None:
        with pytest.raises(ValidationError):
            HardwareProfile(
                target=HardwareTarget.CPU_X86,
                compute_tflops=0.0,
                memory_bandwidth_gbps=50.0,
                memory_capacity_gb=16.0,
            )


class TestModelProfileModel:
    def test_defaults(self) -> None:
        mp = ModelProfile(model_name="test")
        assert mp.parameter_count == 0
        assert mp.batch_size == 1

    def test_notes_field(self) -> None:
        mp = ModelProfile(model_name="test", extra_metadata={"version": "1.0"})
        assert mp.extra_metadata["version"] == "1.0"


class TestOptimizationPlanModel:
    def test_defaults(self) -> None:
        plan = OptimizationPlan(model_name="bert", target_id="a100")
        assert plan.expected_speedup == 1.0
        assert plan.precision_recommendation == "fp32"

    def test_invalid_memory_reduction_above_1(self) -> None:
        with pytest.raises(ValidationError):
            OptimizationPlan(model_name="x", target_id="t4", expected_memory_reduction=1.5)


# ---------------------------------------------------------------------------
# GraphAnalyzer tests
# ---------------------------------------------------------------------------


class TestGraphAnalyzer:
    def test_topological_sort_returns_all_ops(
        self, analyzer: GraphAnalyzer, transformer_graph: ModelGraph
    ) -> None:
        order = analyzer.topological_sort(transformer_graph)
        expected_ids = {op.op_id for op in transformer_graph.operators}
        assert set(order) == expected_ids

    def test_topological_sort_respects_order(
        self, analyzer: GraphAnalyzer, transformer_graph: ModelGraph
    ) -> None:
        order = analyzer.topological_sort(transformer_graph)
        assert order.index("embed") < order.index("attn_0")
        assert order.index("attn_0") < order.index("ln_0")
        assert order.index("ln_0") < order.index("mm_0")

    def test_topological_sort_single_node(self, analyzer: GraphAnalyzer) -> None:
        g = ModelGraph(name="one", operators=[
            GraphOperator(op_id="only", op_type=OperatorType.SOFTMAX)
        ])
        order = analyzer.topological_sort(g)
        assert order == ["only"]

    def test_cycle_raises_value_error(self, analyzer: GraphAnalyzer) -> None:
        ops = [
            GraphOperator(op_id="a", op_type=OperatorType.MATMUL),
            GraphOperator(op_id="b", op_type=OperatorType.MATMUL),
        ]
        g = ModelGraph(name="cycle", operators=ops, edges=[("a", "b"), ("b", "a")])
        with pytest.raises(ValueError, match="cycle"):
            analyzer.topological_sort(g)

    def test_compute_flops_returns_dict(
        self, analyzer: GraphAnalyzer, simple_graph: ModelGraph
    ) -> None:
        flops = analyzer.compute_flops(simple_graph)
        assert isinstance(flops, dict)
        for op in simple_graph.operators:
            assert op.op_id in flops

    def test_compute_flops_positive(
        self, analyzer: GraphAnalyzer, transformer_graph: ModelGraph
    ) -> None:
        flops = analyzer.compute_flops(transformer_graph)
        assert all(v >= 1.0 for v in flops.values())

    def test_identify_bottleneck_returns_op_id(
        self, analyzer: GraphAnalyzer, transformer_graph: ModelGraph
    ) -> None:
        bottleneck = analyzer.identify_bottleneck(transformer_graph, HardwareTarget.A100)
        assert bottleneck is not None
        op_ids = {op.op_id for op in transformer_graph.operators}
        assert bottleneck in op_ids

    def test_identify_bottleneck_empty_graph(self, analyzer: GraphAnalyzer) -> None:
        g = ModelGraph(name="empty")
        bottleneck = analyzer.identify_bottleneck(g, HardwareTarget.CPU_X86)
        assert bottleneck is None

    def test_total_flops_positive(
        self, analyzer: GraphAnalyzer, transformer_graph: ModelGraph
    ) -> None:
        total = analyzer.total_flops(transformer_graph)
        assert total > 0

    def test_total_flops_equals_sum(
        self, analyzer: GraphAnalyzer, simple_graph: ModelGraph
    ) -> None:
        per_op = analyzer.compute_flops(simple_graph)
        total = analyzer.total_flops(simple_graph)
        assert abs(total - sum(per_op.values())) < 1.0


# ---------------------------------------------------------------------------
# HardwareMapper tests
# ---------------------------------------------------------------------------


class TestHardwareMapper:
    def test_get_implementation_cublas_for_cuda(
        self, mapper: HardwareMapper, matmul_op: GraphOperator
    ) -> None:
        impl = mapper.get_implementation(matmul_op, HardwareTarget.GPU_CUDA)
        assert "cublas" in impl

    def test_get_implementation_generic_fallback(
        self, mapper: HardwareMapper
    ) -> None:
        op = GraphOperator(op_id="s", op_type=OperatorType.SOFTMAX)
        impl = mapper.get_implementation(op, HardwareTarget.CPU_ARM)
        assert "softmax" in impl

    def test_estimate_latency_positive(
        self, mapper: HardwareMapper, matmul_op: GraphOperator
    ) -> None:
        latency = mapper.estimate_operator_latency(matmul_op, HardwareTarget.A100)
        assert latency > 0

    def test_estimate_latency_gpu_faster_than_cpu(
        self, mapper: HardwareMapper, matmul_op: GraphOperator
    ) -> None:
        gpu_latency = mapper.estimate_operator_latency(matmul_op, HardwareTarget.A100)
        cpu_latency = mapper.estimate_operator_latency(matmul_op, HardwareTarget.CPU_X86)
        assert gpu_latency < cpu_latency

    def test_estimate_latency_all_targets(
        self, mapper: HardwareMapper, attention_op: GraphOperator
    ) -> None:
        for target in list(HARDWARE_PROFILES.keys())[:5]:
            latency = mapper.estimate_operator_latency(attention_op, target)
            assert latency > 0


# ---------------------------------------------------------------------------
# OptimizationEngine tests
# ---------------------------------------------------------------------------


class TestOptimizationEngine:
    def test_list_passes_all(self, engine: OptimizationEngine) -> None:
        passes = engine.list_passes()
        assert len(passes) > 0
        names = [p.name for p in passes]
        assert "operator_fusion" in names
        assert "constant_folding" in names

    def test_list_passes_filtered_by_target(self, engine: OptimizationEngine) -> None:
        gpu_passes = engine.list_passes(HardwareTarget.GPU_CUDA)
        cpu_passes = engine.list_passes(HardwareTarget.CPU_X86)
        # layout_optimization is GPU-only in the builtin passes
        gpu_names = [p.name for p in gpu_passes]
        assert "layout_optimization" in gpu_names

    def test_optimize_returns_graph_and_passes(
        self, engine: OptimizationEngine, transformer_graph: ModelGraph
    ) -> None:
        new_graph, applied = engine.optimize(transformer_graph, HardwareTarget.GPU_CUDA)
        assert isinstance(new_graph, ModelGraph)
        assert isinstance(applied, list)

    def test_optimize_preserves_ops_unless_fused(
        self, engine: OptimizationEngine, transformer_graph: ModelGraph
    ) -> None:
        new_graph, _ = engine.optimize(transformer_graph, HardwareTarget.CPU_X86)
        # Operators shouldn't disappear entirely
        assert len(new_graph.operators) >= 1

    def test_operator_fusion_fuses_matmul_layernorm(self, engine: OptimizationEngine) -> None:
        ops = [
            GraphOperator(op_id="mm", op_type=OperatorType.MATMUL,
                          input_shapes=[(1, 128, 768)], output_shape=(1, 128, 768)),
            GraphOperator(op_id="ln", op_type=OperatorType.LAYERNORM,
                          input_shapes=[(1, 128, 768)], output_shape=(1, 128, 768)),
        ]
        g = ModelGraph(name="fusable", operators=ops, edges=[("mm", "ln")])
        new_graph, applied = engine.optimize(g, HardwareTarget.GPU_CUDA, passes=["operator_fusion"])
        assert "operator_fusion" in applied
        # Fused op replaces the two
        op_ids = [op.op_id for op in new_graph.operators]
        assert any("fused" in oid for oid in op_ids)

    def test_layout_opt_sets_preferred_layout_cuda(self, engine: OptimizationEngine) -> None:
        ops = [
            GraphOperator(op_id="conv", op_type=OperatorType.CONV2D,
                          input_shapes=[(1, 3, 224, 224)], output_shape=(1, 64, 224, 224))
        ]
        g = ModelGraph(name="conv_model", operators=ops)
        new_graph, _ = engine.optimize(g, HardwareTarget.GPU_CUDA, passes=["layout_optimization"])
        conv_op = new_graph.operator_by_id("conv")
        assert conv_op is not None
        assert conv_op.attributes.get("preferred_layout") == "NHWC"

    def test_specific_pass_selection(
        self, engine: OptimizationEngine, transformer_graph: ModelGraph
    ) -> None:
        _, applied = engine.optimize(
            transformer_graph, HardwareTarget.GPU_CUDA,
            passes=["constant_folding"]
        )
        assert "constant_folding" in applied
        assert "operator_fusion" not in applied


# ---------------------------------------------------------------------------
# PerformancePredictor tests
# ---------------------------------------------------------------------------


class TestPerformancePredictor:
    def test_predict_returns_inference_prediction(
        self, predictor: PerformancePredictor, transformer_graph: ModelGraph
    ) -> None:
        pred = predictor.predict(transformer_graph, HardwareTarget.A100)
        assert isinstance(pred, InferencePrediction)

    def test_predict_total_latency_positive(
        self, predictor: PerformancePredictor, transformer_graph: ModelGraph
    ) -> None:
        pred = predictor.predict(transformer_graph, HardwareTarget.A100)
        assert pred.total_latency_ms > 0

    def test_predict_operator_predictions_count(
        self, predictor: PerformancePredictor, transformer_graph: ModelGraph
    ) -> None:
        pred = predictor.predict(transformer_graph, HardwareTarget.A100)
        assert len(pred.operator_predictions) == len(transformer_graph.operators)

    def test_predict_bottleneck_is_valid_op(
        self, predictor: PerformancePredictor, transformer_graph: ModelGraph
    ) -> None:
        pred = predictor.predict(transformer_graph, HardwareTarget.CPU_X86)
        if pred.bottleneck_op_id:
            op_ids = {op.op_id for op in transformer_graph.operators}
            assert pred.bottleneck_op_id in op_ids

    def test_predict_a100_faster_than_cpu(
        self, predictor: PerformancePredictor, transformer_graph: ModelGraph
    ) -> None:
        pred_gpu = predictor.predict(transformer_graph, HardwareTarget.A100)
        pred_cpu = predictor.predict(transformer_graph, HardwareTarget.CPU_X86)
        assert pred_gpu.total_latency_ms < pred_cpu.total_latency_ms

    def test_predict_empty_graph(self, predictor: PerformancePredictor) -> None:
        g = ModelGraph(name="empty")
        pred = predictor.predict(g, HardwareTarget.CPU_X86)
        assert pred.total_latency_ms == 0.0
        assert pred.operator_predictions == []

    def test_predict_roofline_bound_values(
        self, predictor: PerformancePredictor, simple_graph: ModelGraph
    ) -> None:
        pred = predictor.predict(simple_graph, HardwareTarget.A100)
        for op_pred in pred.operator_predictions:
            assert op_pred.roofline_bound in ("compute", "memory")


# ---------------------------------------------------------------------------
# CrossHardwareComparator tests
# ---------------------------------------------------------------------------


class TestCrossHardwareComparator:
    def test_compare_returns_dict(
        self, comparator: CrossHardwareComparator, transformer_graph: ModelGraph
    ) -> None:
        targets = [HardwareTarget.A100, HardwareTarget.CPU_X86]
        result = comparator.compare(transformer_graph, targets)
        assert isinstance(result, dict)
        assert HardwareTarget.A100 in result

    def test_best_target_in_profiles(
        self, comparator: CrossHardwareComparator, transformer_graph: ModelGraph
    ) -> None:
        targets = [HardwareTarget.A100, HardwareTarget.CPU_X86, HardwareTarget.T4]
        best = comparator.best_target(transformer_graph, targets)
        assert best in targets

    def test_a100_is_best_target(
        self, comparator: CrossHardwareComparator, transformer_graph: ModelGraph
    ) -> None:
        targets = [HardwareTarget.A100, HardwareTarget.CPU_X86, HardwareTarget.CPU_ARM]
        best = comparator.best_target(transformer_graph, targets)
        assert best == HardwareTarget.A100

    def test_speedup_table_returns_dict(
        self, comparator: CrossHardwareComparator, transformer_graph: ModelGraph
    ) -> None:
        table = comparator.speedup_table(transformer_graph, HardwareTarget.CPU_X86)
        assert isinstance(table, dict)
        assert len(table) > 0

    def test_speedup_baseline_equals_one(
        self, comparator: CrossHardwareComparator, transformer_graph: ModelGraph
    ) -> None:
        table = comparator.speedup_table(transformer_graph, HardwareTarget.CPU_X86)
        # Baseline (CPU_X86) speedup vs itself should be 1.0
        assert abs(table.get("cpu_x86", 1.0) - 1.0) < 1e-6

    def test_compare_all_targets_when_none(
        self, comparator: CrossHardwareComparator, simple_graph: ModelGraph
    ) -> None:
        result = comparator.compare(simple_graph)
        assert len(result) == len(HARDWARE_PROFILES)


# ---------------------------------------------------------------------------
# HardwareRegistry tests
# ---------------------------------------------------------------------------


class TestHardwareRegistry:
    def test_get_a100(self, registry: HardwareRegistry) -> None:
        profile = registry.get("a100")
        assert profile.target == HardwareTarget.A100
        assert profile.compute_tflops == 77.6

    def test_get_t4(self, registry: HardwareRegistry) -> None:
        profile = registry.get("t4")
        assert profile.architecture == "turing"

    def test_get_case_insensitive(self, registry: HardwareRegistry) -> None:
        assert registry.get("A100") == registry.get("a100")

    def test_get_unknown_raises_key_error(self, registry: HardwareRegistry) -> None:
        with pytest.raises(KeyError):
            registry.get("unknown_target")

    def test_list_targets_sorted(self, registry: HardwareRegistry) -> None:
        targets = registry.list_targets()
        assert targets == sorted(targets)

    def test_list_targets_includes_all_profiles(self, registry: HardwareRegistry) -> None:
        targets = registry.list_targets()
        for hw in HARDWARE_PROFILES:
            assert hw.value in targets

    def test_register_custom_profile(self, registry: HardwareRegistry) -> None:
        custom = HardwareProfile(
            target=HardwareTarget.NPU,
            compute_tflops=5.0,
            memory_bandwidth_gbps=200.0,
            memory_capacity_gb=4.0,
            architecture="custom_npu_v2",
        )
        registry.register(custom)
        retrieved = registry.get("npu")
        assert retrieved.architecture == "custom_npu_v2"


# ---------------------------------------------------------------------------
# InferenceOptimizer tests
# ---------------------------------------------------------------------------


class TestInferenceOptimizer:
    def test_analyze_returns_optimization_plan(
        self, small_model: ModelProfile
    ) -> None:
        optimizer = InferenceOptimizer()
        plan = optimizer.analyze(small_model, "a100")
        assert isinstance(plan, OptimizationPlan)

    def test_plan_has_techniques(self, small_model: ModelProfile) -> None:
        optimizer = InferenceOptimizer()
        plan = optimizer.analyze(small_model, "a100")
        assert len(plan.recommended_techniques) > 0

    def test_fp16_recommended_for_gpu(self, small_model: ModelProfile) -> None:
        optimizer = InferenceOptimizer()
        plan = optimizer.analyze(small_model, "a100")
        assert "fp16_mixed_precision" in plan.recommended_techniques

    def test_speedup_positive(self, small_model: ModelProfile) -> None:
        optimizer = InferenceOptimizer()
        plan = optimizer.analyze(small_model, "a100")
        assert plan.expected_speedup >= 1.0

    def test_latency_positive(self, small_model: ModelProfile) -> None:
        optimizer = InferenceOptimizer()
        plan = optimizer.analyze(small_model, "a100")
        assert plan.estimated_latency_ms > 0

    def test_throughput_positive(self, small_model: ModelProfile) -> None:
        optimizer = InferenceOptimizer()
        plan = optimizer.analyze(small_model, "a100")
        assert plan.estimated_throughput_qps > 0

    def test_memory_reduction_in_range(self, small_model: ModelProfile) -> None:
        optimizer = InferenceOptimizer()
        plan = optimizer.analyze(small_model, "a100")
        assert 0.0 <= plan.expected_memory_reduction <= 1.0

    def test_model_name_preserved(self, small_model: ModelProfile) -> None:
        optimizer = InferenceOptimizer()
        plan = optimizer.analyze(small_model, "a100")
        assert plan.model_name == small_model.model_name

    def test_target_id_preserved(self, small_model: ModelProfile) -> None:
        optimizer = InferenceOptimizer()
        plan = optimizer.analyze(small_model, "a100")
        assert plan.target_id == "a100"

    def test_kernel_fusion_always_recommended(self, small_model: ModelProfile) -> None:
        optimizer = InferenceOptimizer()
        plan = optimizer.analyze(small_model, "cpu_x86")
        assert "kernel_fusion" in plan.recommended_techniques


# ---------------------------------------------------------------------------
# CrossCompiler tests
# ---------------------------------------------------------------------------


class TestCrossCompiler:
    def test_compile_returns_dict(self, small_model: ModelProfile) -> None:
        compiler = CrossCompiler()
        artifact = compiler.compile(small_model, "a100")
        assert isinstance(artifact, dict)

    def test_compile_required_keys(self, small_model: ModelProfile) -> None:
        compiler = CrossCompiler()
        artifact = compiler.compile(small_model, "a100")
        required = {"model_name", "target", "export_command", "compile_flags",
                    "runtime_requirements", "memory_required_gb", "fits_in_device_memory"}
        assert required.issubset(artifact.keys())

    def test_compile_a100_has_fp16_flag(self, small_model: ModelProfile) -> None:
        compiler = CrossCompiler()
        artifact = compiler.compile(small_model, "a100")
        assert "--fp16" in artifact["compile_flags"]

    def test_compile_model_name_present(self, small_model: ModelProfile) -> None:
        compiler = CrossCompiler()
        artifact = compiler.compile(small_model, "a100")
        assert artifact["model_name"] == small_model.model_name

    def test_compile_fits_in_memory_for_small_model(self, small_model: ModelProfile) -> None:
        compiler = CrossCompiler()
        artifact = compiler.compile(small_model, "a100")
        assert artifact["fits_in_device_memory"] is True

    def test_compile_precision_support_keys(self, small_model: ModelProfile) -> None:
        compiler = CrossCompiler()
        artifact = compiler.compile(small_model, "a100")
        precision = artifact["precision_support"]
        assert isinstance(precision, dict)
        assert "fp32" in precision
        assert "fp16" in precision
        assert "int8" in precision
        assert "tensor_cores" in precision

    def test_compile_cpu_has_onnx_command(self, small_model: ModelProfile) -> None:
        compiler = CrossCompiler()
        artifact = compiler.compile(small_model, "cpu_x86")
        assert "onnx" in artifact["export_command"].lower() or "onnx" in str(artifact["export_command"]).lower()

    def test_compile_runtime_requirements_list(self, small_model: ModelProfile) -> None:
        compiler = CrossCompiler()
        artifact = compiler.compile(small_model, "a100")
        reqs = artifact["runtime_requirements"]
        assert isinstance(reqs, list)
        assert len(reqs) > 0

    def test_compile_large_model_not_fitting(self) -> None:
        large_model = ModelProfile(
            model_name="llama-70b",
            parameter_count=70_000_000_000,
            flops_per_inference=1e15,
            memory_footprint_mb=140_000.0,  # 140 GB, won't fit in T4 (16 GB)
        )
        compiler = CrossCompiler()
        artifact = compiler.compile(large_model, "t4")
        assert artifact["fits_in_device_memory"] is False


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


@given(target=st.sampled_from(list(HARDWARE_PROFILES.keys())))
@settings(max_examples=11, deadline=5000)
def test_hardware_profile_compute_positive(target: HardwareTarget) -> None:
    profile = HARDWARE_PROFILES[target]
    assert profile.compute_tflops > 0
    assert profile.memory_bandwidth_gbps > 0
    assert profile.memory_capacity_gb > 0


@given(
    num_ops=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=20, deadline=5000)
def test_topological_sort_linear_chain(num_ops: int) -> None:
    ops = [
        GraphOperator(op_id=f"op_{i}", op_type=OperatorType.MATMUL,
                      input_shapes=[(1, 4)], output_shape=(1, 4))
        for i in range(num_ops)
    ]
    edges = [(f"op_{i}", f"op_{i + 1}") for i in range(num_ops - 1)]
    g = ModelGraph(name="chain", operators=ops, edges=edges)
    analyzer = GraphAnalyzer()
    order = analyzer.topological_sort(g)
    assert order == [f"op_{i}" for i in range(num_ops)]
