"""CLI entry point for aumai-chipbridge.

Commands:
    analyze  -- analyze a model graph from JSON
    optimize -- apply optimization passes to a model graph
    predict  -- predict performance on a target hardware
    compare  -- compare performance across hardware targets
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .core import CrossHardwareComparator, GraphAnalyzer, OptimizationEngine, PerformancePredictor
from .models import GraphOperator, HardwareTarget, ModelGraph, OperatorType


def _load_graph(path: Path) -> ModelGraph:
    """Load a ModelGraph from a JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ModelGraph.model_validate(data)
    except Exception as exc:
        click.echo(f"ERROR loading graph from {path}: {exc}", err=True)
        sys.exit(1)


def _demo_graph() -> ModelGraph:
    """Create a small demo transformer-like graph for testing."""
    ops = [
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
    return ModelGraph(name="demo_transformer", operators=ops, edges=edges)


@click.group()
@click.version_option()
def main() -> None:
    """AumAI ChipBridge -- cross-hardware ML inference optimization CLI."""


@main.command("analyze")
@click.option("--graph", "graph_path", default=None, type=click.Path(path_type=Path),
              help="Path to ModelGraph JSON. Uses built-in demo if omitted.")
@click.option("--target", default="gpu_cuda",
              type=click.Choice([t.value for t in HardwareTarget], case_sensitive=False),
              help="Hardware target for bottleneck analysis.")
def analyze_command(graph_path: Path | None, target: str) -> None:
    """Analyze a model graph: FLOPs, topological order, and bottleneck operator.

    Example:

        aumai-chipbridge analyze --target gpu_cuda
        aumai-chipbridge analyze --graph model.json --target cpu_x86
    """
    graph = _load_graph(graph_path) if graph_path else _demo_graph()
    hardware_target = HardwareTarget(target)
    analyzer = GraphAnalyzer()

    try:
        topo_order = analyzer.topological_sort(graph)
    except ValueError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    flops_map = analyzer.compute_flops(graph)
    total_flops = analyzer.total_flops(graph)
    bottleneck = analyzer.identify_bottleneck(graph, hardware_target)

    click.echo(f"\nModel Graph: {graph.name}")
    click.echo(f"Operators  : {len(graph.operators)}")
    click.echo(f"Edges      : {len(graph.edges)}")
    click.echo(f"Total FLOPs: {total_flops:,.0f}")
    click.echo(f"Target     : {target}")
    click.echo(f"Bottleneck : {bottleneck}")

    click.echo("\nExecution Order (topological):")
    for op_id in topo_order:
        flops = flops_map.get(op_id, 0)
        marker = " <-- BOTTLENECK" if op_id == bottleneck else ""
        click.echo(f"  {op_id:20s}  {flops:>15,.0f} FLOPs{marker}")


@main.command("optimize")
@click.option("--graph", "graph_path", default=None, type=click.Path(path_type=Path))
@click.option("--target", default="gpu_cuda",
              type=click.Choice([t.value for t in HardwareTarget], case_sensitive=False))
@click.option("--passes", default=None, help="Comma-separated list of pass names to apply.")
@click.option("--output", "output_path", default=None, type=click.Path(path_type=Path))
def optimize_command(
    graph_path: Path | None,
    target: str,
    passes: str | None,
    output_path: Path | None,
) -> None:
    """Apply optimization passes to a model graph.

    Example:

        aumai-chipbridge optimize --target gpu_cuda --passes operator_fusion,layout_optimization
    """
    graph = _load_graph(graph_path) if graph_path else _demo_graph()
    hardware_target = HardwareTarget(target)
    engine = OptimizationEngine()

    pass_names = [p.strip() for p in passes.split(",")] if passes else None

    click.echo(f"\nOptimizing graph '{graph.name}' for {target}...")
    click.echo(f"Original operators: {len(graph.operators)}")

    optimized, applied = engine.optimize(graph, hardware_target, passes=pass_names)

    click.echo(f"Optimized operators: {len(optimized.operators)}")
    click.echo(f"Applied passes: {', '.join(applied) if applied else 'none'}")

    if output_path:
        output_path.write_text(optimized.model_dump_json(indent=2), encoding="utf-8")
        click.echo(f"Optimized graph saved to {output_path}")
    else:
        click.echo("\nOptimized operators:")
        for op in optimized.operators:
            click.echo(f"  {op.op_id:25s} [{op.op_type.value}]")


@main.command("predict")
@click.option("--graph", "graph_path", default=None, type=click.Path(path_type=Path))
@click.option("--target", default="gpu_cuda",
              type=click.Choice([t.value for t in HardwareTarget], case_sensitive=False))
def predict_command(graph_path: Path | None, target: str) -> None:
    """Predict inference latency on a target hardware.

    Example:

        aumai-chipbridge predict --target tpu
    """
    graph = _load_graph(graph_path) if graph_path else _demo_graph()
    hardware_target = HardwareTarget(target)
    predictor = PerformancePredictor()
    prediction = predictor.predict(graph, hardware_target)

    click.echo(f"\nPerformance Prediction: {graph.name} on {target}")
    click.echo(f"Total latency: {prediction.total_latency_ms:.4f} ms")
    click.echo(f"Bottleneck   : {prediction.bottleneck_op_id}")

    click.echo("\nPer-Operator Breakdown:")
    click.echo(f"  {'Op ID':<25} {'Latency (ms)':>14} {'Bound':>10} {'FLOPs':>15}")
    click.echo("  " + "-" * 68)
    for op_pred in sorted(prediction.operator_predictions, key=lambda p: -p.latency_ms):
        click.echo(
            f"  {op_pred.op_id:<25} {op_pred.latency_ms:>14.6f} {op_pred.roofline_bound:>10} {op_pred.flops:>15,.0f}"
        )


@main.command("compare")
@click.option("--graph", "graph_path", default=None, type=click.Path(path_type=Path))
@click.option("--baseline", default="cpu_x86",
              type=click.Choice([t.value for t in HardwareTarget], case_sensitive=False))
def compare_command(graph_path: Path | None, baseline: str) -> None:
    """Compare model performance across all hardware targets.

    Example:

        aumai-chipbridge compare
        aumai-chipbridge compare --baseline cpu_arm
    """
    graph = _load_graph(graph_path) if graph_path else _demo_graph()
    baseline_target = HardwareTarget(baseline)
    comparator = CrossHardwareComparator()

    best = comparator.best_target(graph)
    speedups = comparator.speedup_table(graph, baseline=baseline_target)
    comparison = comparator.compare(graph)

    click.echo(f"\nCross-Hardware Comparison: {graph.name}")
    click.echo(f"Baseline: {baseline}")
    click.echo(f"Best target: {best.value}")

    click.echo(f"\n  {'Target':<15} {'Latency (ms)':>14} {'Speedup vs baseline':>20}")
    click.echo("  " + "-" * 52)

    for target in sorted(comparison.keys(), key=lambda t: comparison[t].total_latency_ms):
        pred = comparison[target]
        speedup = speedups.get(target.value, 1.0)
        marker = " (*)" if target == best else ""
        click.echo(
            f"  {target.value:<15} {pred.total_latency_ms:>14.4f} {speedup:>20.2f}x{marker}"
        )


if __name__ == "__main__":
    main()
