"""Core logic for aumai-chipbridge — cross-hardware ML inference optimization.

Provides:
- GraphAnalyzer: Topological sort, per-operator FLOPs, bottleneck identification.
- HardwareMapper: Map operators to hardware implementations, estimate timing.
- OptimizationEngine: Apply graph optimization passes.
- PerformancePredictor: Predict inference latency analytically.
- CrossHardwareComparator: Compare performance across hardware targets.
- HardwareRegistry: Pre-built profiles for A100, T4, M2, Xeon, ARM.
- InferenceOptimizer: Recommend optimization plans for a model on a target.
- CrossCompiler: Generate deployment artefact descriptions.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Optional

from .models import (
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
# Hardware profiles database (base set)
# ---------------------------------------------------------------------------

HARDWARE_PROFILES: dict[HardwareTarget, HardwareProfile] = {
    HardwareTarget.CPU_X86: HardwareProfile(
        target=HardwareTarget.CPU_X86,
        compute_tflops=0.5,
        memory_bandwidth_gbps=50.0,
        memory_capacity_gb=16.0,
        has_tensor_cores=False,
        supports_fp16=False,
        supports_int8=True,
        architecture="x86_64",
    ),
    HardwareTarget.CPU_ARM: HardwareProfile(
        target=HardwareTarget.CPU_ARM,
        compute_tflops=0.3,
        memory_bandwidth_gbps=40.0,
        memory_capacity_gb=8.0,
        has_tensor_cores=False,
        supports_fp16=True,
        supports_int8=True,
        architecture="arm64",
    ),
    HardwareTarget.GPU_CUDA: HardwareProfile(
        target=HardwareTarget.GPU_CUDA,
        compute_tflops=10.0,
        memory_bandwidth_gbps=900.0,
        memory_capacity_gb=24.0,
        has_tensor_cores=True,
        supports_fp16=True,
        supports_int8=True,
        architecture="sm_86",
    ),
    HardwareTarget.GPU_ROCM: HardwareProfile(
        target=HardwareTarget.GPU_ROCM,
        compute_tflops=8.0,
        memory_bandwidth_gbps=700.0,
        memory_capacity_gb=16.0,
        has_tensor_cores=True,
        supports_fp16=True,
        supports_int8=True,
        architecture="gfx908",
    ),
    HardwareTarget.TPU: HardwareProfile(
        target=HardwareTarget.TPU,
        compute_tflops=45.0,
        memory_bandwidth_gbps=600.0,
        memory_capacity_gb=8.0,
        has_tensor_cores=True,
        supports_fp16=True,
        supports_int8=True,
        architecture="tpu_v4",
    ),
    HardwareTarget.NPU: HardwareProfile(
        target=HardwareTarget.NPU,
        compute_tflops=2.0,
        memory_bandwidth_gbps=100.0,
        memory_capacity_gb=2.0,
        has_tensor_cores=False,
        supports_fp16=True,
        supports_int8=True,
        architecture="generic_npu",
    ),
    # Named hardware targets
    HardwareTarget.A100: HardwareProfile(
        target=HardwareTarget.A100,
        compute_tflops=77.6,       # FP32 tensor core
        memory_bandwidth_gbps=2000.0,
        memory_capacity_gb=80.0,
        has_tensor_cores=True,
        supports_fp16=True,
        supports_int8=True,
        architecture="ampere",
        notes="NVIDIA A100 80GB HBM2e",
    ),
    HardwareTarget.T4: HardwareProfile(
        target=HardwareTarget.T4,
        compute_tflops=8.1,
        memory_bandwidth_gbps=300.0,
        memory_capacity_gb=16.0,
        has_tensor_cores=True,
        supports_fp16=True,
        supports_int8=True,
        architecture="turing",
        notes="NVIDIA T4 16GB GDDR6",
    ),
    HardwareTarget.M2: HardwareProfile(
        target=HardwareTarget.M2,
        compute_tflops=3.6,
        memory_bandwidth_gbps=100.0,
        memory_capacity_gb=24.0,
        has_tensor_cores=False,
        supports_fp16=True,
        supports_int8=True,
        architecture="apple_silicon",
        notes="Apple M2 with unified memory",
    ),
    HardwareTarget.XEON: HardwareProfile(
        target=HardwareTarget.XEON,
        compute_tflops=1.0,
        memory_bandwidth_gbps=204.8,
        memory_capacity_gb=512.0,
        has_tensor_cores=False,
        supports_fp16=False,
        supports_int8=True,
        architecture="x86_64_avx512",
        notes="Intel Xeon Scalable with AVX-512",
    ),
    HardwareTarget.ARM: HardwareProfile(
        target=HardwareTarget.ARM,
        compute_tflops=0.5,
        memory_bandwidth_gbps=51.2,
        memory_capacity_gb=16.0,
        has_tensor_cores=False,
        supports_fp16=True,
        supports_int8=True,
        architecture="arm64_neon",
        notes="ARM Cortex-A78 with NEON SIMD",
    ),
}


def _compute_operator_flops(op: GraphOperator) -> float:
    """Estimate FLOPs for a single operator based on type and shapes."""
    shape = op.input_shapes[0] if op.input_shapes else ()

    if op.op_type == OperatorType.MATMUL:
        m = shape[0] if len(shape) >= 1 else 1
        k = shape[1] if len(shape) >= 2 else 1
        n = op.output_shape[1] if len(op.output_shape) >= 2 else k
        return float(2 * m * k * n)

    elif op.op_type == OperatorType.CONV2D:
        batch = shape[0] if len(shape) >= 1 else 1
        channels_in = shape[1] if len(shape) >= 2 else 1
        height = shape[2] if len(shape) >= 3 else 1
        width = shape[3] if len(shape) >= 4 else 1
        channels_out = op.attributes.get("out_channels", channels_in)
        kernel_h = op.attributes.get("kernel_h", 3)
        kernel_w = op.attributes.get("kernel_w", 3)
        return float(2 * batch * channels_out * height * width * channels_in * kernel_h * kernel_w)

    elif op.op_type == OperatorType.ATTENTION:
        batch = shape[0] if len(shape) >= 1 else 1
        seq_len = shape[1] if len(shape) >= 2 else 1
        d_model = shape[2] if len(shape) >= 3 else 1
        return float(4 * batch * seq_len * seq_len * d_model)

    elif op.op_type in (OperatorType.LAYERNORM, OperatorType.SOFTMAX, OperatorType.GELU):
        total = 1
        for d in shape:
            total *= d
        return float(5 * total)

    elif op.op_type == OperatorType.EMBEDDING:
        batch = shape[0] if len(shape) >= 1 else 1
        seq_len = shape[1] if len(shape) >= 2 else 1
        embed_dim = op.attributes.get("embed_dim", 768)
        return float(batch * seq_len * embed_dim)

    return 1.0


def _memory_bytes_for_operator(op: GraphOperator) -> float:
    """Estimate memory traffic for an operator (float32, 4 bytes/element)."""
    total_input_elements = 0
    for shape in op.input_shapes:
        elements = 1
        for d in shape:
            elements *= d
        total_input_elements += elements

    output_elements = 1
    for d in op.output_shape:
        output_elements *= d

    return float((total_input_elements + output_elements) * 4)


# ---------------------------------------------------------------------------
# GraphAnalyzer
# ---------------------------------------------------------------------------


class GraphAnalyzer:
    """Analyze a model computation graph for performance characteristics."""

    def topological_sort(self, graph: ModelGraph) -> list[str]:
        """Return operator IDs in topological execution order (Kahn's algorithm)."""
        in_degree: dict[str, int] = {op.op_id: 0 for op in graph.operators}
        adjacency: dict[str, list[str]] = defaultdict(list)

        for source, dest in graph.edges:
            adjacency[source].append(dest)
            in_degree[dest] = in_degree.get(dest, 0) + 1

        queue: deque[str] = deque(
            op_id for op_id, degree in in_degree.items() if degree == 0
        )
        sorted_ops: list[str] = []

        while queue:
            node = queue.popleft()
            sorted_ops.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_ops) != len(graph.operators):
            raise ValueError("Graph contains a cycle — cannot perform topological sort.")

        return sorted_ops

    def compute_flops(self, graph: ModelGraph) -> dict[str, float]:
        """Compute estimated FLOPs for each operator."""
        return {op.op_id: _compute_operator_flops(op) for op in graph.operators}

    def identify_bottleneck(
        self, graph: ModelGraph, target: HardwareTarget
    ) -> Optional[str]:
        """Identify the operator with the highest predicted latency."""
        profile = HARDWARE_PROFILES.get(target)
        if not profile or not graph.operators:
            return None

        max_latency = -1.0
        bottleneck_id: Optional[str] = None

        for op in graph.operators:
            flops = _compute_operator_flops(op)
            mem_bytes = _memory_bytes_for_operator(op)
            ai = flops / mem_bytes if mem_bytes > 0 else 0.0
            ridge = (profile.compute_tflops * 1e12) / (profile.memory_bandwidth_gbps * 1e9)

            if ai >= ridge:
                bound_seconds = flops / (profile.compute_tflops * 1e12)
            else:
                bound_seconds = mem_bytes / (profile.memory_bandwidth_gbps * 1e9)

            latency_ms = bound_seconds * 1000
            if latency_ms > max_latency:
                max_latency = latency_ms
                bottleneck_id = op.op_id

        return bottleneck_id

    def total_flops(self, graph: ModelGraph) -> float:
        """Sum FLOPs across all operators."""
        return sum(_compute_operator_flops(op) for op in graph.operators)


# ---------------------------------------------------------------------------
# HardwareMapper
# ---------------------------------------------------------------------------


class HardwareMapper:
    """Map operators to hardware-specific implementations and estimate timing."""

    _SPEEDUP_TABLE: dict[HardwareTarget, dict[OperatorType, float]] = {
        HardwareTarget.CPU_X86: {t: 1.0 for t in OperatorType},
        HardwareTarget.CPU_ARM: {
            OperatorType.MATMUL: 0.8, OperatorType.CONV2D: 0.9,
            OperatorType.ATTENTION: 0.75, OperatorType.LAYERNORM: 0.9,
            OperatorType.SOFTMAX: 0.9, OperatorType.GELU: 0.85,
            OperatorType.EMBEDDING: 1.1,
        },
        HardwareTarget.GPU_CUDA: {
            OperatorType.MATMUL: 20.0, OperatorType.CONV2D: 25.0,
            OperatorType.ATTENTION: 18.0, OperatorType.LAYERNORM: 12.0,
            OperatorType.SOFTMAX: 15.0, OperatorType.GELU: 14.0,
            OperatorType.EMBEDDING: 5.0,
        },
        HardwareTarget.GPU_ROCM: {
            OperatorType.MATMUL: 16.0, OperatorType.CONV2D: 20.0,
            OperatorType.ATTENTION: 14.0, OperatorType.LAYERNORM: 10.0,
            OperatorType.SOFTMAX: 12.0, OperatorType.GELU: 11.0,
            OperatorType.EMBEDDING: 4.0,
        },
        HardwareTarget.TPU: {
            OperatorType.MATMUL: 90.0, OperatorType.CONV2D: 80.0,
            OperatorType.ATTENTION: 70.0, OperatorType.LAYERNORM: 30.0,
            OperatorType.SOFTMAX: 25.0, OperatorType.GELU: 20.0,
            OperatorType.EMBEDDING: 10.0,
        },
        HardwareTarget.NPU: {
            OperatorType.MATMUL: 4.0, OperatorType.CONV2D: 6.0,
            OperatorType.ATTENTION: 3.0, OperatorType.LAYERNORM: 2.0,
            OperatorType.SOFTMAX: 2.5, OperatorType.GELU: 2.0,
            OperatorType.EMBEDDING: 2.0,
        },
        HardwareTarget.A100: {
            OperatorType.MATMUL: 155.0, OperatorType.CONV2D: 140.0,
            OperatorType.ATTENTION: 120.0, OperatorType.LAYERNORM: 50.0,
            OperatorType.SOFTMAX: 60.0, OperatorType.GELU: 55.0,
            OperatorType.EMBEDDING: 15.0,
        },
        HardwareTarget.T4: {
            OperatorType.MATMUL: 16.0, OperatorType.CONV2D: 18.0,
            OperatorType.ATTENTION: 14.0, OperatorType.LAYERNORM: 9.0,
            OperatorType.SOFTMAX: 11.0, OperatorType.GELU: 10.0,
            OperatorType.EMBEDDING: 4.0,
        },
        HardwareTarget.M2: {
            OperatorType.MATMUL: 7.0, OperatorType.CONV2D: 9.0,
            OperatorType.ATTENTION: 6.0, OperatorType.LAYERNORM: 4.0,
            OperatorType.SOFTMAX: 5.0, OperatorType.GELU: 4.5,
            OperatorType.EMBEDDING: 2.5,
        },
        HardwareTarget.XEON: {
            OperatorType.MATMUL: 2.0, OperatorType.CONV2D: 2.2,
            OperatorType.ATTENTION: 1.8, OperatorType.LAYERNORM: 1.5,
            OperatorType.SOFTMAX: 1.6, OperatorType.GELU: 1.4,
            OperatorType.EMBEDDING: 1.2,
        },
        HardwareTarget.ARM: {
            OperatorType.MATMUL: 1.0, OperatorType.CONV2D: 1.1,
            OperatorType.ATTENTION: 0.9, OperatorType.LAYERNORM: 0.9,
            OperatorType.SOFTMAX: 1.0, OperatorType.GELU: 0.85,
            OperatorType.EMBEDDING: 1.0,
        },
    }

    def get_implementation(self, op: GraphOperator, target: HardwareTarget) -> str:
        """Return the recommended implementation name for an operator on a target."""
        impl_map: dict[tuple[OperatorType, HardwareTarget], str] = {
            (OperatorType.MATMUL, HardwareTarget.GPU_CUDA): "cublas_gemm",
            (OperatorType.MATMUL, HardwareTarget.A100): "cublas_tensor_core_gemm",
            (OperatorType.MATMUL, HardwareTarget.T4): "cublas_int8_gemm",
            (OperatorType.MATMUL, HardwareTarget.TPU): "tpu_dot_general",
            (OperatorType.MATMUL, HardwareTarget.CPU_X86): "mkl_sgemm",
            (OperatorType.MATMUL, HardwareTarget.XEON): "mkl_avx512_sgemm",
            (OperatorType.MATMUL, HardwareTarget.ARM): "arm_compute_lib_gemm",
            (OperatorType.MATMUL, HardwareTarget.M2): "accelerate_blas_gemm",
            (OperatorType.CONV2D, HardwareTarget.GPU_CUDA): "cudnn_conv2d",
            (OperatorType.CONV2D, HardwareTarget.A100): "cudnn_amp_conv2d",
            (OperatorType.CONV2D, HardwareTarget.CPU_X86): "mkl_conv2d",
            (OperatorType.ATTENTION, HardwareTarget.GPU_CUDA): "flash_attention_v2",
            (OperatorType.ATTENTION, HardwareTarget.A100): "flash_attention_v2_fp16",
            (OperatorType.ATTENTION, HardwareTarget.TPU): "tpu_attention",
        }
        return impl_map.get((op.op_type, target), f"generic_{op.op_type.value}")

    def estimate_operator_latency(
        self, op: GraphOperator, target: HardwareTarget
    ) -> float:
        """Estimate operator execution time in milliseconds."""
        profile = HARDWARE_PROFILES.get(target)
        if not profile:
            return 0.001

        flops = _compute_operator_flops(op)
        mem_bytes = _memory_bytes_for_operator(op)

        compute_seconds = flops / (profile.compute_tflops * 1e12)
        memory_seconds = mem_bytes / (profile.memory_bandwidth_gbps * 1e9)
        raw_latency_s = max(compute_seconds, memory_seconds)

        speedup = self._SPEEDUP_TABLE.get(target, {}).get(op.op_type, 1.0)
        adjusted_latency_ms = (raw_latency_s / speedup) * 1000

        return max(0.001, adjusted_latency_ms)


# ---------------------------------------------------------------------------
# OptimizationEngine
# ---------------------------------------------------------------------------

_BUILTIN_PASSES: list[OptimizationPass] = [
    OptimizationPass(
        name="operator_fusion",
        description="Fuse consecutive operators to reduce memory traffic and kernel launch overhead.",
        applicable_targets=list(HardwareTarget),
    ),
    OptimizationPass(
        name="constant_folding",
        description="Evaluate operators with compile-time constant inputs and inline results.",
        applicable_targets=list(HardwareTarget),
    ),
    OptimizationPass(
        name="dead_code_elimination",
        description="Remove operators whose outputs are never consumed.",
        applicable_targets=list(HardwareTarget),
    ),
    OptimizationPass(
        name="layout_optimization",
        description="Rewrite tensor layouts (NCHW <-> NHWC) to match hardware-preferred formats.",
        applicable_targets=[
            HardwareTarget.GPU_CUDA, HardwareTarget.GPU_ROCM,
            HardwareTarget.A100, HardwareTarget.T4, HardwareTarget.TPU,
        ],
    ),
]


class OptimizationEngine:
    """Apply graph optimization passes to a model computation graph."""

    def list_passes(self, target: Optional[HardwareTarget] = None) -> list[OptimizationPass]:
        """List available optimization passes, optionally filtered by target."""
        if target is None:
            return _BUILTIN_PASSES
        return [p for p in _BUILTIN_PASSES if target in p.applicable_targets]

    def optimize(
        self,
        graph: ModelGraph,
        target: HardwareTarget,
        passes: Optional[list[str]] = None,
    ) -> tuple[ModelGraph, list[str]]:
        """Apply optimization passes to the graph."""
        applicable = self.list_passes(target)
        if passes is not None:
            applicable = [p for p in applicable if p.name in passes]

        current_graph = graph
        applied_passes: list[str] = []

        for pass_obj in applicable:
            result = self._apply_pass(current_graph, pass_obj, target)
            if result is not None:
                current_graph = result
                applied_passes.append(pass_obj.name)

        return current_graph, applied_passes

    def _apply_pass(
        self, graph: ModelGraph, pass_obj: OptimizationPass, target: HardwareTarget
    ) -> Optional[ModelGraph]:
        if pass_obj.name == "operator_fusion":
            return self._fuse_operators(graph)
        elif pass_obj.name == "constant_folding":
            return self._constant_folding(graph)
        elif pass_obj.name == "dead_code_elimination":
            return self._dead_code_elimination(graph)
        elif pass_obj.name == "layout_optimization":
            return self._layout_optimization(graph, target)
        return None

    def _fuse_operators(self, graph: ModelGraph) -> ModelGraph:
        ops_by_id = {op.op_id: op for op in graph.operators}
        fused_ids: set[str] = set()
        result_ops: list[GraphOperator] = []

        for edge_src, edge_dst in graph.edges:
            src_op = ops_by_id.get(edge_src)
            dst_op = ops_by_id.get(edge_dst)
            if (
                src_op and dst_op
                and src_op.op_type == OperatorType.MATMUL
                and dst_op.op_type == OperatorType.LAYERNORM
                and edge_src not in fused_ids
                and edge_dst not in fused_ids
            ):
                fused_op = GraphOperator(
                    op_id=f"fused_{edge_src}_{edge_dst}",
                    op_type=OperatorType.MATMUL,
                    input_shapes=src_op.input_shapes,
                    output_shape=dst_op.output_shape,
                    attributes={**src_op.attributes, **dst_op.attributes, "fused": True},
                )
                fused_ids.add(edge_src)
                fused_ids.add(edge_dst)
                result_ops.append(fused_op)

        for op in graph.operators:
            if op.op_id not in fused_ids:
                result_ops.append(op)

        kept_edges = [
            (s, d) for s, d in graph.edges
            if not (s in fused_ids and d in fused_ids)
        ]
        return ModelGraph(name=graph.name, operators=result_ops, edges=kept_edges)

    def _constant_folding(self, graph: ModelGraph) -> ModelGraph:
        consumed = {dst for _, dst in graph.edges}
        producing = {src for src, _ in graph.edges}
        isolated_ids = {
            op.op_id for op in graph.operators
            if op.op_id not in consumed and op.op_id not in producing
            and len(graph.operators) > 1
        }
        new_ops = [op for op in graph.operators if op.op_id not in isolated_ids]
        new_edges = [(s, d) for s, d in graph.edges if s not in isolated_ids and d not in isolated_ids]
        return ModelGraph(name=graph.name, operators=new_ops, edges=new_edges)

    def _dead_code_elimination(self, graph: ModelGraph) -> ModelGraph:
        consumed = {dst for _, dst in graph.edges}
        producing = {src for src, _ in graph.edges}
        all_ids = {op.op_id for op in graph.operators}
        terminal_candidates = all_ids - consumed
        dead_ids = terminal_candidates - producing
        if len(dead_ids) >= len(graph.operators):
            dead_ids = set()
        new_ops = [op for op in graph.operators if op.op_id not in dead_ids]
        new_edges = [(s, d) for s, d in graph.edges if s not in dead_ids and d not in dead_ids]
        return ModelGraph(name=graph.name, operators=new_ops, edges=new_edges)

    def _layout_optimization(self, graph: ModelGraph, target: HardwareTarget) -> ModelGraph:
        preferred_layout = "NHWC" if target in (
            HardwareTarget.GPU_CUDA, HardwareTarget.A100, HardwareTarget.T4, HardwareTarget.TPU
        ) else "NCHW"
        new_ops = []
        for op in graph.operators:
            if op.op_type == OperatorType.CONV2D:
                new_op = GraphOperator(
                    op_id=op.op_id,
                    op_type=op.op_type,
                    input_shapes=op.input_shapes,
                    output_shape=op.output_shape,
                    attributes={**op.attributes, "preferred_layout": preferred_layout},
                )
                new_ops.append(new_op)
            else:
                new_ops.append(op)
        return ModelGraph(name=graph.name, operators=new_ops, edges=graph.edges)


# ---------------------------------------------------------------------------
# PerformancePredictor
# ---------------------------------------------------------------------------


class PerformancePredictor:
    """Predict inference latency on a target hardware using the roofline model."""

    def __init__(self) -> None:
        self._mapper = HardwareMapper()

    def predict(self, graph: ModelGraph, target: HardwareTarget) -> InferencePrediction:
        """Predict end-to-end inference latency."""
        profile = HARDWARE_PROFILES.get(target)
        if not profile:
            return InferencePrediction(model_name=graph.name, target=target, total_latency_ms=0.0)

        op_predictions: list[OperatorPrediction] = []
        total_latency = 0.0
        max_latency = -1.0
        bottleneck_id: Optional[str] = None

        for op in graph.operators:
            flops = _compute_operator_flops(op)
            mem_bytes = _memory_bytes_for_operator(op)
            ai = flops / mem_bytes if mem_bytes > 0 else 0.0
            ridge = (profile.compute_tflops * 1e12) / (profile.memory_bandwidth_gbps * 1e9)

            roofline_bound = "compute" if ai >= ridge else "memory"
            latency_ms = self._mapper.estimate_operator_latency(op, target)
            total_latency += latency_ms

            if latency_ms > max_latency:
                max_latency = latency_ms
                bottleneck_id = op.op_id

            op_predictions.append(
                OperatorPrediction(
                    op_id=op.op_id,
                    target=target,
                    latency_ms=round(latency_ms, 6),
                    roofline_bound=roofline_bound,
                    flops=flops,
                )
            )

        return InferencePrediction(
            model_name=graph.name,
            target=target,
            total_latency_ms=round(total_latency, 4),
            operator_predictions=op_predictions,
            bottleneck_op_id=bottleneck_id,
        )


# ---------------------------------------------------------------------------
# CrossHardwareComparator
# ---------------------------------------------------------------------------


class CrossHardwareComparator:
    """Compare model inference performance across multiple hardware targets."""

    def __init__(self) -> None:
        self._predictor = PerformancePredictor()

    def compare(
        self,
        graph: ModelGraph,
        targets: Optional[list[HardwareTarget]] = None,
    ) -> dict[HardwareTarget, InferencePrediction]:
        """Compare model performance across hardware targets."""
        if targets is None:
            targets = list(HARDWARE_PROFILES.keys())
        return {target: self._predictor.predict(graph, target) for target in targets}

    def best_target(
        self,
        graph: ModelGraph,
        targets: Optional[list[HardwareTarget]] = None,
    ) -> HardwareTarget:
        """Identify the hardware target with the lowest predicted latency."""
        comparison = self.compare(graph, targets)
        return min(comparison.keys(), key=lambda t: comparison[t].total_latency_ms)

    def speedup_table(
        self, graph: ModelGraph, baseline: HardwareTarget = HardwareTarget.CPU_X86
    ) -> dict[str, float]:
        """Compute speedup relative to a baseline target."""
        comparison = self.compare(graph)
        baseline_latency = comparison[baseline].total_latency_ms
        if baseline_latency <= 0:
            return {}
        return {
            target.value: round(baseline_latency / pred.total_latency_ms, 2)
            for target, pred in comparison.items()
            if pred.total_latency_ms > 0
        }


# ---------------------------------------------------------------------------
# HardwareRegistry
# ---------------------------------------------------------------------------


class HardwareRegistry:
    """Registry of pre-built hardware target profiles.

    Includes profiles for: A100, T4, M2, Xeon, ARM, GPU_CUDA, CPU_X86,
    CPU_ARM, TPU, NPU, GPU_ROCM.

    Example:
        >>> registry = HardwareRegistry()
        >>> profile = registry.get("a100")
        >>> all_targets = registry.list_targets()
    """

    def __init__(self) -> None:
        self._profiles: dict[str, HardwareProfile] = {
            target.value: profile for target, profile in HARDWARE_PROFILES.items()
        }

    def get(self, target_id: str) -> HardwareProfile:
        """Retrieve a hardware profile by target ID string.

        Args:
            target_id: Case-insensitive target ID (e.g. "a100", "t4").

        Returns:
            HardwareProfile for the target.

        Raises:
            KeyError: If no profile exists for the given ID.
        """
        key = target_id.lower()
        if key not in self._profiles:
            raise KeyError(
                f"No hardware profile for '{target_id}'. "
                f"Available: {', '.join(sorted(self._profiles.keys()))}"
            )
        return self._profiles[key]

    def list_targets(self) -> list[str]:
        """Return all registered target IDs."""
        return sorted(self._profiles.keys())

    def register(self, profile: HardwareProfile) -> None:
        """Register a custom hardware profile.

        Args:
            profile: HardwareProfile to add to the registry.
        """
        self._profiles[profile.target.value] = profile


# ---------------------------------------------------------------------------
# InferenceOptimizer
# ---------------------------------------------------------------------------


class InferenceOptimizer:
    """Recommend optimization plans for deploying a model on a hardware target.

    Analyses the ModelProfile and HardwareProfile to recommend:
    - Precision reduction (FP16/BF16/INT8)
    - Flash Attention if model has attention layers
    - Tensor cores if hardware supports them
    - INT8 quantization for edge/mobile targets
    - Memory mapping for large models

    Example:
        >>> registry = HardwareRegistry()
        >>> optimizer = InferenceOptimizer(registry)
        >>> plan = optimizer.analyze(model_profile, "a100")
    """

    def __init__(self, registry: Optional[HardwareRegistry] = None) -> None:
        self._registry = registry or HardwareRegistry()

    def analyze(self, model: ModelProfile, target_id: str) -> OptimizationPlan:
        """Generate an optimization plan for the model on the target hardware.

        Args:
            model: The model's computational profile.
            target_id: Hardware target identifier.

        Returns:
            OptimizationPlan with recommended techniques and projections.
        """
        hw = self._registry.get(target_id)
        techniques: list[str] = []
        warnings: list[str] = []

        # Precision recommendations
        if hw.supports_fp16 and hw.has_tensor_cores:
            techniques.append("fp16_mixed_precision")
            precision_rec = "fp16"
            precision_speedup = 2.0
        elif hw.supports_fp16:
            techniques.append("fp16_mixed_precision")
            precision_rec = "fp16"
            precision_speedup = 1.5
        else:
            precision_rec = "fp32"
            precision_speedup = 1.0

        # Flash Attention for models with attention on supported GPU targets only
        if model.has_attention and hw.target in (
            HardwareTarget.GPU_CUDA, HardwareTarget.A100, HardwareTarget.T4, HardwareTarget.GPU_ROCM
        ):
            if hw.supports_fp16:
                techniques.append("flash_attention")

        # INT8 quantization for memory-constrained or edge targets
        if hw.supports_int8:
            if model.memory_footprint_mb > hw.memory_capacity_gb * 1024 * 0.7:
                techniques.append("int8_quantization")
                warnings.append("Model may not fit in device memory; INT8 quantization strongly recommended.")
            else:
                techniques.append("int8_quantization")

        # Kernel fusion
        techniques.append("kernel_fusion")

        # Operator fusion
        techniques.append("operator_fusion")

        # Estimate performance
        baseline_latency_ms = (
            model.flops_per_inference / (hw.compute_tflops * 1e12) * 1000
            if hw.compute_tflops > 0 and model.flops_per_inference > 0
            else 10.0
        )
        optimized_latency = baseline_latency_ms / precision_speedup
        throughput_qps = 1000.0 / optimized_latency if optimized_latency > 0 else 0.0

        # Memory reduction from FP16/INT8
        memory_reduction = 0.5 if "fp16_mixed_precision" in techniques else 0.0
        if "int8_quantization" in techniques:
            memory_reduction = max(memory_reduction, 0.75)

        return OptimizationPlan(
            model_name=model.model_name,
            target_id=target_id,
            recommended_techniques=techniques,
            expected_speedup=round(precision_speedup, 2),
            expected_memory_reduction=round(memory_reduction, 2),
            estimated_latency_ms=round(optimized_latency, 4),
            estimated_throughput_qps=round(throughput_qps, 2),
            precision_recommendation=precision_rec,
            warnings=warnings,
            notes=(
                f"Based on roofline model: {hw.compute_tflops:.1f} TFLOPS FP32, "
                f"{hw.memory_bandwidth_gbps:.0f} GB/s bandwidth."
            ),
        )


# ---------------------------------------------------------------------------
# CrossCompiler
# ---------------------------------------------------------------------------


class CrossCompiler:
    """Generate deployment artefact descriptions for cross-hardware compilation.

    Produces a structured description of compile flags, runtime requirements,
    and framework-specific export commands for deploying a model on a target.

    Example:
        >>> registry = HardwareRegistry()
        >>> compiler = CrossCompiler(registry)
        >>> artefact = compiler.compile(model_profile, "a100")
        >>> print(artefact["export_command"])
    """

    def __init__(self, registry: Optional[HardwareRegistry] = None) -> None:
        self._registry = registry or HardwareRegistry()

    def compile(self, model: ModelProfile, target_id: str) -> dict[str, object]:
        """Generate a deployment artefact description.

        Args:
            model: The model profile to compile for.
            target_id: Hardware target identifier.

        Returns:
            Dictionary with compilation instructions, flags, and commands.
        """
        hw = self._registry.get(target_id)
        target = hw.target

        framework_commands: dict[HardwareTarget, str] = {
            HardwareTarget.GPU_CUDA: (
                f"torch.export.export(model, ...) && "
                f"torch_tensorrt.compile(model, inputs=[...], enabled_precisions={{torch.float16}})"
            ),
            HardwareTarget.A100: (
                f"torch_tensorrt.compile(model, inputs=[...], "
                f"enabled_precisions={{torch.float16, torch.int8}}, "
                f"use_python_runtime=False)"
            ),
            HardwareTarget.T4: (
                f"torch_tensorrt.compile(model, inputs=[...], "
                f"enabled_precisions={{torch.int8}}, calib_dataset=calib_loader)"
            ),
            HardwareTarget.TPU: (
                f"import torch_xla.core.xla_model as xm; "
                f"model = xm.to_xla_device(model); "
                f"torch.jit.script(model)"
            ),
            HardwareTarget.M2: (
                f"coremltools.convert(model, minimum_deployment_target=coremltools.target.macOS14)"
            ),
            HardwareTarget.XEON: (
                f"torch.onnx.export(model, dummy_input, 'model.onnx'); "
                f"optimum-cli export openvino --model . --task text-generation"
            ),
            HardwareTarget.ARM: (
                f"tflite_convert --saved_model_dir=. --output_file=model.tflite "
                f"--target_spec.supported_ops=TFLITE_BUILTINS_INT8"
            ),
            HardwareTarget.CPU_X86: (
                f"torch.onnx.export(model, dummy_input, 'model.onnx'); "
                f"onnxruntime.InferenceSession('model.onnx')"
            ),
            HardwareTarget.CPU_ARM: (
                f"tflite_convert --saved_model_dir=. --output_file=model.tflite"
            ),
            HardwareTarget.NPU: (
                f"npu_toolkit.compile(model, target_device='npu', quant_config=int8_config)"
            ),
        }

        compile_flags: dict[HardwareTarget, list[str]] = {
            HardwareTarget.A100: ["--fp16", "--int8", "--gpu-architecture=sm_80", "--optimize-all"],
            HardwareTarget.T4: ["--int8", "--gpu-architecture=sm_75", "--optimize-all"],
            HardwareTarget.GPU_CUDA: ["--fp16", "--gpu-architecture=sm_86"],
            HardwareTarget.XEON: ["--avx512", "--mkl-dnn", "--openmp"],
            HardwareTarget.ARM: ["--neon", "--fp16", "--armv8a"],
            HardwareTarget.M2: ["--ane-target", "--fp16"],
            HardwareTarget.CPU_X86: ["--avx2", "--openmp"],
            HardwareTarget.CPU_ARM: ["--neon", "--fp16"],
        }

        return {
            "model_name": model.model_name,
            "target": target.value,
            "architecture": hw.architecture,
            "export_command": framework_commands.get(target, f"# No specific export command for {target_id}"),
            "compile_flags": compile_flags.get(target, []),
            "runtime_requirements": self._runtime_requirements(target),
            "memory_required_gb": round(model.memory_footprint_mb / 1024, 2),
            "device_memory_gb": hw.memory_capacity_gb,
            "fits_in_device_memory": (model.memory_footprint_mb / 1024) <= hw.memory_capacity_gb,
            "precision_support": {
                "fp32": True,
                "fp16": hw.supports_fp16,
                "int8": hw.supports_int8,
                "tensor_cores": hw.has_tensor_cores,
            },
        }

    def _runtime_requirements(self, target: HardwareTarget) -> list[str]:
        """Return runtime library requirements for the target."""
        reqs: dict[HardwareTarget, list[str]] = {
            HardwareTarget.A100: ["cuda>=11.8", "cudnn>=8.6", "tensorrt>=8.5"],
            HardwareTarget.T4: ["cuda>=11.0", "cudnn>=8.0", "tensorrt>=8.0"],
            HardwareTarget.GPU_CUDA: ["cuda>=11.0", "cudnn>=8.0"],
            HardwareTarget.TPU: ["jax>=0.4", "torch-xla>=2.0"],
            HardwareTarget.M2: ["coremltools>=7.0", "macos>=13.0"],
            HardwareTarget.XEON: ["openvino>=2023.0", "mkl>=2022"],
            HardwareTarget.ARM: ["tflite-runtime>=2.13", "armnn>=23.0"],
            HardwareTarget.CPU_X86: ["onnxruntime>=1.16"],
            HardwareTarget.CPU_ARM: ["onnxruntime>=1.16"],
            HardwareTarget.NPU: ["npu-toolkit>=1.0"],
        }
        return reqs.get(target, ["onnxruntime>=1.16"])


__all__ = [
    "HARDWARE_PROFILES",
    "GraphAnalyzer",
    "HardwareMapper",
    "OptimizationEngine",
    "PerformancePredictor",
    "CrossHardwareComparator",
    "HardwareRegistry",
    "InferenceOptimizer",
    "CrossCompiler",
]
