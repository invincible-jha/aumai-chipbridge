"""AumAI ChipBridge — quickstart examples.

Demonstrates five core workflows:
  1. Build a model graph and analyze it (topology, FLOPs, bottleneck)
  2. Predict inference latency and compare hardware targets
  3. Apply graph optimization passes
  4. Generate an InferenceOptimizer plan for a model profile
  5. Generate cross-compilation artefact descriptions

Run this file directly to verify your installation:

    python examples/quickstart.py
"""

from __future__ import annotations

from aumai_chipbridge.core import (
    CrossCompiler,
    CrossHardwareComparator,
    GraphAnalyzer,
    HardwareRegistry,
    InferenceOptimizer,
    OptimizationEngine,
    PerformancePredictor,
)
from aumai_chipbridge.models import (
    GraphOperator,
    HardwareTarget,
    ModelGraph,
    ModelProfile,
    OperatorType,
)


# ---------------------------------------------------------------------------
# Shared helper — build a single-layer transformer block
# ---------------------------------------------------------------------------


def build_transformer_graph() -> ModelGraph:
    """Build a single-layer transformer block as a ModelGraph.

    Graph topology:
        embed -> attn_0 -> norm_0 -> ff_0 -> gelu_0 -> ff_1 -> norm_1 -> softmax_out
    """
    operators = [
        GraphOperator(
            op_id="embed",
            op_type=OperatorType.EMBEDDING,
            input_shapes=[(1, 128)],
            output_shape=(1, 128, 768),
            attributes={"embed_dim": 768, "vocab_size": 32000},
        ),
        GraphOperator(
            op_id="attn_0",
            op_type=OperatorType.ATTENTION,
            input_shapes=[(1, 128, 768)],
            output_shape=(1, 128, 768),
            attributes={"num_heads": 12},
        ),
        GraphOperator(
            op_id="norm_0",
            op_type=OperatorType.LAYERNORM,
            input_shapes=[(1, 128, 768)],
            output_shape=(1, 128, 768),
        ),
        GraphOperator(
            op_id="ff_0",
            op_type=OperatorType.MATMUL,
            input_shapes=[(128, 768)],
            output_shape=(128, 3072),
        ),
        GraphOperator(
            op_id="gelu_0",
            op_type=OperatorType.GELU,
            input_shapes=[(128, 3072)],
            output_shape=(128, 3072),
        ),
        GraphOperator(
            op_id="ff_1",
            op_type=OperatorType.MATMUL,
            input_shapes=[(128, 3072)],
            output_shape=(128, 768),
        ),
        GraphOperator(
            op_id="norm_1",
            op_type=OperatorType.LAYERNORM,
            input_shapes=[(128, 768)],
            output_shape=(128, 768),
        ),
        GraphOperator(
            op_id="softmax_out",
            op_type=OperatorType.SOFTMAX,
            input_shapes=[(128, 768)],
            output_shape=(128, 32000),
        ),
    ]
    edges = [
        ("embed", "attn_0"),
        ("attn_0", "norm_0"),
        ("norm_0", "ff_0"),
        ("ff_0", "gelu_0"),
        ("gelu_0", "ff_1"),
        ("ff_1", "norm_1"),
        ("norm_1", "softmax_out"),
    ]
    return ModelGraph(name="transformer_block", operators=operators, edges=edges)


# ---------------------------------------------------------------------------
# Demo 1: Graph analysis
# ---------------------------------------------------------------------------


def demo_graph_analysis() -> None:
    """Analyze a transformer graph: topological sort, FLOPs, bottleneck.

    GraphAnalyzer performs three analytical operations:
    - topological_sort: valid linear execution order using Kahn's algorithm
    - compute_flops: per-operator FLOPs estimates based on operator type and shapes
    - identify_bottleneck: the slowest operator under the roofline model for a target
    """
    print("\n" + "=" * 60)
    print("DEMO 1: Graph Analysis")
    print("=" * 60)

    graph = build_transformer_graph()
    analyzer = GraphAnalyzer()
    target = HardwareTarget.GPU_CUDA

    # Topological sort — Kahn's algorithm, raises ValueError on cycles
    execution_order = analyzer.topological_sort(graph)
    flops_map = analyzer.compute_flops(graph)
    total_flops = analyzer.total_flops(graph)
    bottleneck = analyzer.identify_bottleneck(graph, target)

    print(f"\nModel:      {graph.name}")
    print(f"Operators:  {len(graph.operators)}")
    print(f"Edges:      {len(graph.edges)}")
    print(f"Total FLOPs:{total_flops:>20,.0f}")
    print(f"Target:     {target.value}")
    print(f"Bottleneck: {bottleneck}")

    print("\nExecution order and per-operator FLOPs:")
    for op_id in execution_order:
        flops = flops_map.get(op_id, 0.0)
        marker = "  <-- BOTTLENECK" if op_id == bottleneck else ""
        print(f"  {op_id:<22}  {flops:>16,.0f} FLOPs{marker}")


# ---------------------------------------------------------------------------
# Demo 2: Performance prediction and cross-hardware comparison
# ---------------------------------------------------------------------------


def demo_performance_prediction() -> None:
    """Predict latency on multiple targets and rank them.

    PerformancePredictor applies the roofline model per operator:
    - Operators above the ridge point are compute-bound
    - Operators below are memory-bound
    CrossHardwareComparator runs predictions across all built-in profiles.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Performance Prediction and Cross-Hardware Comparison")
    print("=" * 60)

    graph = build_transformer_graph()
    predictor = PerformancePredictor()
    comparator = CrossHardwareComparator()

    # Detailed per-operator prediction on A100
    target = HardwareTarget.A100
    prediction = predictor.predict(graph, target)

    print(f"\nDetailed prediction on {target.value}:")
    print(f"  Total latency:    {prediction.total_latency_ms:.4f} ms")
    print(f"  Bottleneck op:    {prediction.bottleneck_op_id}")
    print(f"\n  {'Operator':<22} {'Latency (ms)':>14} {'Bound':>10} {'FLOPs':>16}")
    print("  " + "-" * 66)
    for op_pred in sorted(prediction.operator_predictions, key=lambda p: -p.latency_ms):
        print(
            f"  {op_pred.op_id:<22} {op_pred.latency_ms:>14.6f}"
            f" {op_pred.roofline_bound:>10} {op_pred.flops:>16,.0f}"
        )

    # Cross-hardware speedup table relative to CPU x86 baseline
    baseline = HardwareTarget.CPU_X86
    speedups = comparator.speedup_table(graph, baseline=baseline)
    comparison = comparator.compare(graph)
    best = comparator.best_target(graph)

    print(f"\nCross-hardware comparison (baseline = {baseline.value}):")
    print(f"  Best target: {best.value}")
    print(f"\n  {'Target':<15} {'Latency (ms)':>14} {'Speedup':>10}")
    print("  " + "-" * 43)
    for hw_target in sorted(comparison.keys(), key=lambda t: comparison[t].total_latency_ms):
        pred = comparison[hw_target]
        speedup = speedups.get(hw_target.value, 1.0)
        marker = " (*)" if hw_target == best else ""
        print(f"  {hw_target.value:<15} {pred.total_latency_ms:>14.4f} {speedup:>9.2f}x{marker}")


# ---------------------------------------------------------------------------
# Demo 3: Graph optimization passes
# ---------------------------------------------------------------------------


def demo_optimization_passes() -> None:
    """Apply optimization passes and compare latency before and after.

    The four built-in passes:
    - operator_fusion:       fuse consecutive MATMUL+LAYERNORM pairs
    - constant_folding:      remove isolated constant operators
    - dead_code_elimination: remove terminal unused operators
    - layout_optimization:   set preferred NHWC/NCHW on CONV2D operators
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Optimization Passes")
    print("=" * 60)

    graph = build_transformer_graph()
    engine = OptimizationEngine()
    predictor = PerformancePredictor()
    target = HardwareTarget.A100

    # List available passes for this target
    available = engine.list_passes(target)
    print(f"\nAvailable passes for {target.value} ({len(available)} total):")
    for p in available:
        print(f"  {p.name:<30}  {p.description[:55]}...")

    # Baseline latency
    baseline_pred = predictor.predict(graph, target)

    # Apply all applicable passes
    optimized_graph, applied = engine.optimize(graph, target)
    optimized_pred = predictor.predict(optimized_graph, target)

    print(f"\nBefore: {len(graph.operators)} operators, "
          f"{baseline_pred.total_latency_ms:.4f} ms on {target.value}")
    print(f"After:  {len(optimized_graph.operators)} operators, "
          f"{optimized_pred.total_latency_ms:.4f} ms on {target.value}")
    print(f"Passes applied: {applied}")

    print(f"\nOptimized operator list:")
    for op in optimized_graph.operators:
        tags: list[str] = []
        if op.attributes.get("fused"):
            tags.append("fused")
        layout = op.attributes.get("preferred_layout", "")
        if layout:
            tags.append(f"layout={layout}")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        print(f"  {op.op_id:<30} [{op.op_type.value}]{tag_str}")

    # Apply only a specific subset of passes
    partial_graph, partial_applied = engine.optimize(
        graph, target, passes=["operator_fusion"]
    )
    print(f"\nWith only operator_fusion: "
          f"{len(partial_graph.operators)} operators, passes={partial_applied}")


# ---------------------------------------------------------------------------
# Demo 4: InferenceOptimizer — model-level optimization planning
# ---------------------------------------------------------------------------


def demo_inference_optimizer() -> None:
    """Generate optimization plans for a BERT-base model profile on several targets.

    InferenceOptimizer works at the ModelProfile level (parameter count, FLOPs,
    memory) rather than the graph level. It recommends precision, quantization,
    flash attention, and other deployment techniques.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: InferenceOptimizer — Model-Level Optimization Plans")
    print("=" * 60)

    bert_base = ModelProfile(
        model_name="bert-base-uncased",
        parameter_count=110_000_000,
        flops_per_inference=22_500_000_000,    # 22.5 GFLOPs
        memory_footprint_mb=440.0,
        batch_size=1,
        has_attention=True,
        has_convolutions=False,
    )

    optimizer = InferenceOptimizer()
    target_ids = ["a100", "t4", "m2", "xeon", "cpu_x86"]

    print(f"\nModel: {bert_base.model_name}")
    print(f"  Parameters:     {bert_base.parameter_count:,}")
    print(f"  FLOPs/inf:      {bert_base.flops_per_inference:,.0f}")
    print(f"  Memory:         {bert_base.memory_footprint_mb:.0f} MB")
    print(f"  Has attention:  {bert_base.has_attention}")

    print(f"\n  {'Target':<10} {'Precision':>10} {'Speedup':>8} {'Latency ms':>12} {'QPS':>8}")
    print("  " + "-" * 54)

    for target_id in target_ids:
        plan = optimizer.analyze(bert_base, target_id)
        print(
            f"  {target_id:<10} {plan.precision_recommendation:>10}"
            f" {plan.expected_speedup:>7.1f}x"
            f" {plan.estimated_latency_ms:>12.4f}"
            f" {plan.estimated_throughput_qps:>8.1f}"
        )
        for warning in plan.warnings:
            print(f"  WARNING ({target_id}): {warning}")

    # Show full detail for A100
    print("\nFull plan for a100:")
    plan = optimizer.analyze(bert_base, "a100")
    print(f"  Techniques:         {plan.recommended_techniques}")
    print(f"  Memory reduction:   {plan.expected_memory_reduction * 100:.0f}%")
    print(f"  Notes:              {plan.notes}")


# ---------------------------------------------------------------------------
# Demo 5: CrossCompiler — deployment artefact descriptions
# ---------------------------------------------------------------------------


def demo_cross_compiler() -> None:
    """Generate deployment artefact descriptions for a model on multiple targets.

    CrossCompiler provides export commands, compile flags, runtime dependencies,
    and a memory-fit check for each hardware target — no actual compilation runs.
    """
    print("\n" + "=" * 60)
    print("DEMO 5: CrossCompiler — Deployment Artefact Descriptions")
    print("=" * 60)

    # GPT-2 medium — large enough that smaller devices may not fit it
    gpt2_medium = ModelProfile(
        model_name="gpt2-medium",
        parameter_count=345_000_000,
        flops_per_inference=3_400_000_000_000,    # 3.4 TFLOPs
        memory_footprint_mb=1380.0,
        batch_size=1,
        has_attention=True,
        has_convolutions=False,
    )

    compiler = CrossCompiler()
    registry = HardwareRegistry()

    print(f"\nModel: {gpt2_medium.model_name}")
    print(f"  Size: {gpt2_medium.memory_footprint_mb:.0f} MB "
          f"({gpt2_medium.memory_footprint_mb / 1024:.2f} GB)")

    for target_id in ["a100", "t4", "m2", "xeon"]:
        artefact = compiler.compile(gpt2_medium, target_id)
        hw = registry.get(target_id)
        fits = artefact["fits_in_device_memory"]
        fits_label = "fits" if fits else "DOES NOT FIT — quantize first"

        print(f"\n--- {target_id.upper()} ({hw.architecture}) ---")
        print(f"  Memory: {artefact['memory_required_gb']:.2f} GB / "
              f"{artefact['device_memory_gb']} GB ({fits_label})")
        precision = artefact["precision_support"]
        print(f"  Precision: fp16={precision['fp16']}, "
              f"int8={precision['int8']}, "
              f"tensor_cores={precision['tensor_cores']}")
        print(f"  Compile flags: {artefact['compile_flags']}")
        print(f"  Runtime deps:  {artefact['runtime_requirements']}")
        export_cmd = str(artefact["export_command"])
        # Truncate long commands for display
        truncated = export_cmd[:88] + "..." if len(export_cmd) > 88 else export_cmd
        print(f"  Export cmd:    {truncated}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all five AumAI ChipBridge quickstart demonstrations."""
    print("AumAI ChipBridge — Quickstart Demonstrations")
    print("Cross-hardware ML inference optimization (analytical, no GPU required)")

    demo_graph_analysis()
    demo_performance_prediction()
    demo_optimization_passes()
    demo_inference_optimizer()
    demo_cross_compiler()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
    print("\nNext steps:")
    print("  aumai-chipbridge --help")
    print("  aumai-chipbridge compare")
    print("  aumai-chipbridge predict --target a100")
    print("  Read docs/api-reference.md for the complete API surface.")


if __name__ == "__main__":
    main()
