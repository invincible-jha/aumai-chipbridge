# Getting Started with aumai-chipbridge

Cross-hardware ML inference optimization.

---

## Prerequisites

- **Python 3.11 or newer** — ChipBridge uses modern type hint syntax and relies on Python 3.11+ standard library features
- **pip** — for installation (or your preferred package manager: uv, poetry, pipenv)
- **No GPU required** — ChipBridge predicts performance analytically; you do not need access to any of the hardware targets to use it

### Optional but recommended

- `ipython` or `jupyter` — for interactive exploration of predictions and comparisons
- `jq` — for inspecting the JSON graph files that ChipBridge reads and writes

---

## Installation

### From PyPI

```bash
pip install aumai-chipbridge
```

### From source (for development)

```bash
git clone https://github.com/aumai/aumai-chipbridge
cd aumai-chipbridge
pip install -e ".[dev]"
```

### Verify installation

```bash
aumai-chipbridge --version
python -c "import aumai_chipbridge; print(aumai_chipbridge.__version__)"
```

Both commands should print `0.1.0` (or the current version).

---

## Your First Commands

ChipBridge ships with a built-in demo transformer graph so you can explore immediately without creating any files.

### Step 1 — Analyze the demo graph

```bash
aumai-chipbridge analyze --target gpu_cuda
```

This prints the operators in topological order, their estimated FLOPs, and which one is the bottleneck on a CUDA GPU.

### Step 2 — Predict latency

```bash
aumai-chipbridge predict --target tpu
```

This predicts the per-operator latency on a TPU, with a compute vs. memory bound classification for each operator.

### Step 3 — Compare all hardware targets

```bash
aumai-chipbridge compare
```

This compares every registered hardware target and prints a speedup table relative to the CPU x86 baseline.

### Step 4 — Optimize the graph

```bash
aumai-chipbridge optimize --target a100 --passes operator_fusion,layout_optimization
```

This applies two optimization passes and prints the resulting operator list.

---

## Step-by-Step Tutorial

This tutorial walks through the full ChipBridge workflow: build a graph, analyze it, optimize it, compare hardware, and generate deployment artefacts.

### Step 1 — Define Your Model Graph

Create a file called `my_model.json`:

```json
{
  "name": "small_transformer",
  "operators": [
    {
      "op_id": "embed",
      "op_type": "embedding",
      "input_shapes": [[1, 64]],
      "output_shape": [1, 64, 512],
      "attributes": {"embed_dim": 512, "vocab_size": 32000}
    },
    {
      "op_id": "attn_0",
      "op_type": "attention",
      "input_shapes": [[1, 64, 512]],
      "output_shape": [1, 64, 512],
      "attributes": {"num_heads": 8}
    },
    {
      "op_id": "norm_0",
      "op_type": "layernorm",
      "input_shapes": [[1, 64, 512]],
      "output_shape": [1, 64, 512]
    },
    {
      "op_id": "ff_0",
      "op_type": "matmul",
      "input_shapes": [[64, 512]],
      "output_shape": [64, 2048]
    },
    {
      "op_id": "gelu_0",
      "op_type": "gelu",
      "input_shapes": [[64, 2048]],
      "output_shape": [64, 2048]
    },
    {
      "op_id": "ff_1",
      "op_type": "matmul",
      "input_shapes": [[64, 2048]],
      "output_shape": [64, 512]
    }
  ],
  "edges": [
    ["embed", "attn_0"],
    ["attn_0", "norm_0"],
    ["norm_0", "ff_0"],
    ["ff_0", "gelu_0"],
    ["gelu_0", "ff_1"]
  ]
}
```

### Step 2 — Analyze the Graph

```bash
aumai-chipbridge analyze --graph my_model.json --target a100
```

Note the bottleneck operator. For transformer-like models, the attention operator almost always dominates because its FLOPs grow quadratically with sequence length.

### Step 3 — Predict Latency on Multiple Targets

```bash
aumai-chipbridge predict --graph my_model.json --target a100
aumai-chipbridge predict --graph my_model.json --target t4
aumai-chipbridge predict --graph my_model.json --target m2
aumai-chipbridge predict --graph my_model.json --target xeon
```

### Step 4 — Compare All Targets

```bash
aumai-chipbridge compare --graph my_model.json --baseline cpu_x86
```

This gives you a ranked list of all targets by latency, plus speedup relative to your baseline.

### Step 5 — Optimize the Graph

Apply all applicable optimization passes for A100:

```bash
aumai-chipbridge optimize --graph my_model.json --target a100 --output optimized.json
```

Or apply specific passes only:

```bash
aumai-chipbridge optimize --graph my_model.json --target a100 \
  --passes operator_fusion,layout_optimization --output optimized.json
```

### Step 6 — Re-predict on the Optimized Graph

```bash
aumai-chipbridge predict --graph optimized.json --target a100
```

Compare the latency figures before and after optimization.

### Step 7 — Generate an Optimization Plan in Python

The CLI focuses on graph-level analysis. For model-level planning (precision, quantization, framework export), use the Python API:

```python
from aumai_chipbridge.core import InferenceOptimizer
from aumai_chipbridge.models import ModelProfile

model = ModelProfile(
    model_name="small_transformer",
    parameter_count=117_000_000,
    flops_per_inference=2.4e12,
    memory_footprint_mb=450,
    has_attention=True,
    has_convolutions=False,
)

optimizer = InferenceOptimizer()
plan = optimizer.analyze(model, "a100")

print(f"Recommended precision: {plan.precision_recommendation}")
print(f"Techniques: {plan.recommended_techniques}")
print(f"Expected speedup: {plan.expected_speedup}x")
print(f"Estimated latency: {plan.estimated_latency_ms:.2f} ms")
print(f"Est. throughput: {plan.estimated_throughput_qps:.1f} QPS")
```

### Step 8 — Generate Cross-Compilation Artefacts

```python
from aumai_chipbridge.core import CrossCompiler
from aumai_chipbridge.models import ModelProfile

model = ModelProfile(
    model_name="small_transformer",
    parameter_count=117_000_000,
    flops_per_inference=2.4e12,
    memory_footprint_mb=450,
    has_attention=True,
)

compiler = CrossCompiler()

for target_id in ["a100", "m2", "xeon"]:
    artefact = compiler.compile(model, target_id)
    print(f"\n=== {target_id.upper()} ===")
    print(f"  Export: {artefact['export_command']}")
    print(f"  Flags:  {artefact['compile_flags']}")
    print(f"  Deps:   {artefact['runtime_requirements']}")
```

---

## 5 Common Patterns

### Pattern 1 — Find the Best Hardware for Your Model

```python
from aumai_chipbridge import (
    ModelGraph, GraphOperator, OperatorType,
    CrossHardwareComparator, HardwareTarget,
)

graph = ModelGraph(
    name="my_model",
    operators=[
        GraphOperator(
            op_id="main_attn",
            op_type=OperatorType.ATTENTION,
            input_shapes=[(4, 512, 1024)],
            output_shape=(4, 512, 1024),
            attributes={"num_heads": 16},
        ),
    ],
    edges=[],
)

comparator = CrossHardwareComparator()
best = comparator.best_target(graph)
print(f"Optimal deployment hardware: {best.value}")
```

### Pattern 2 — Compare Cloud vs. Edge Targets

```python
from aumai_chipbridge import CrossHardwareComparator, HardwareTarget

comparator = CrossHardwareComparator()

cloud_targets = [HardwareTarget.A100, HardwareTarget.T4, HardwareTarget.GPU_CUDA]
edge_targets = [HardwareTarget.M2, HardwareTarget.ARM, HardwareTarget.NPU]

cloud_comparison = comparator.compare(graph, targets=cloud_targets)
edge_comparison = comparator.compare(graph, targets=edge_targets)

print("Cloud targets:")
for target, pred in cloud_comparison.items():
    print(f"  {target.value}: {pred.total_latency_ms:.4f} ms")

print("Edge targets:")
for target, pred in edge_comparison.items():
    print(f"  {target.value}: {pred.total_latency_ms:.4f} ms")
```

### Pattern 3 — Identify and Investigate the Bottleneck

```python
from aumai_chipbridge import GraphAnalyzer, PerformancePredictor, HardwareTarget

analyzer = GraphAnalyzer()
predictor = PerformancePredictor()
target = HardwareTarget.A100

bottleneck_id = analyzer.identify_bottleneck(graph, target)
print(f"Bottleneck operator: {bottleneck_id}")

pred = predictor.predict(graph, target)
for op_pred in pred.operator_predictions:
    if op_pred.op_id == bottleneck_id:
        print(f"  Latency: {op_pred.latency_ms:.4f} ms")
        print(f"  Bound: {op_pred.roofline_bound}")
        print(f"  FLOPs: {op_pred.flops:,.0f}")
```

### Pattern 4 — Apply Selective Optimization Passes

```python
from aumai_chipbridge import OptimizationEngine, PerformancePredictor, HardwareTarget

engine = OptimizationEngine()
predictor = PerformancePredictor()
target = HardwareTarget.A100

baseline_pred = predictor.predict(graph, target)

fused_graph, _ = engine.optimize(graph, target, passes=["operator_fusion"])
fused_pred = predictor.predict(fused_graph, target)

full_optimized, applied = engine.optimize(graph, target)
full_pred = predictor.predict(full_optimized, target)

print(f"Baseline:                  {baseline_pred.total_latency_ms:.4f} ms")
print(f"After fusion only:         {fused_pred.total_latency_ms:.4f} ms")
print(f"All passes {applied}: {full_pred.total_latency_ms:.4f} ms")
```

### Pattern 5 — Register and Use Custom Hardware

```python
from aumai_chipbridge.core import HardwareRegistry, InferenceOptimizer
from aumai_chipbridge.models import HardwareProfile, HardwareTarget, ModelProfile

registry = HardwareRegistry()
registry.register(HardwareProfile(
    target=HardwareTarget.NPU,
    compute_tflops=12.0,
    memory_bandwidth_gbps=400.0,
    memory_capacity_gb=16.0,
    has_tensor_cores=True,
    supports_fp16=True,
    supports_int8=True,
    architecture="custom_npu_v4",
    notes="Next-gen internal NPU with INT8 matrix cores",
))

optimizer = InferenceOptimizer(registry=registry)
model = ModelProfile(
    model_name="edge_model",
    parameter_count=10_000_000,
    flops_per_inference=500e9,
    memory_footprint_mb=40,
    has_attention=True,
)
plan = optimizer.analyze(model, "npu")
print(plan.recommended_techniques)
```

---

## Troubleshooting FAQ

**Q: The CLI says `aumai-chipbridge: command not found`. What do I do?**

A: The CLI entry point is installed into your Python environment's `bin/` (or `Scripts/` on Windows) directory. Make sure that directory is on your PATH. If you installed with `pip install --user`, add `~/.local/bin` to your PATH. Alternatively, run `python -m aumai_chipbridge.cli` as a fallback.

---

**Q: I get `KeyError: No hardware profile for 'my_chip'`. How do I add my hardware?**

A: Use `HardwareRegistry.register()` with a custom `HardwareProfile`. See Pattern 5 above. You will need to pick the closest `HardwareTarget` enum member (e.g., `HardwareTarget.NPU`) since the enum values are fixed. The registry key is derived from `profile.target.value`.

---

**Q: Why does the latency prediction differ significantly from my real hardware benchmark?**

A: ChipBridge uses an analytical roofline model based on peak hardware specifications. Real latency is affected by factors not modeled here: driver overhead, kernel launch latency, memory fragmentation, thermal throttling, batch composition, and OS scheduling. Predictions are most accurate for large, compute-bound operations (big matrix multiplies) and least accurate for tiny memory-bound operations. Always validate with proper hardware benchmarking tools such as `torch.cuda.synchronize()` + timing loops, `nsys`, or `perf`.

---

**Q: I have a cyclic graph. The `topological_sort` raises a `ValueError`. Is this a bug?**

A: Cyclic graphs — for example, RNNs with recurrent connections — cannot be topologically sorted because there is no valid linear execution order. ChipBridge expects acyclic DAGs (directed acyclic graphs). For RNNs, unroll the graph by repeating operators for each time step, treating each step as a separate set of nodes.

---

**Q: Can I load a PyTorch model directly instead of writing JSON by hand?**

A: ChipBridge does not have a PyTorch importer in the current release. To bridge the gap, use `torch.fx.symbolic_trace` or `torch.onnx.export` to extract the computation graph, then parse the operator nodes into `GraphOperator` objects. This is intentionally left to the user because operator naming conventions vary significantly across frameworks.

---

**Q: The `operator_fusion` pass did not reduce my operator count. Why?**

A: The current `operator_fusion` pass specifically targets adjacent MATMUL + LAYERNORM pairs in the edge list. If your graph has different adjacencies, the pass will not find fusable pairs. Future releases will expand the fusion pattern set.

---

**Q: What is the difference between `InferenceOptimizer` and `OptimizationEngine`?**

A: They operate at different levels. `OptimizationEngine` works on a `ModelGraph` — the low-level computation graph of operators and edges. It transforms the graph structure itself. `InferenceOptimizer` works on a `ModelProfile` — a high-level description of a model's parameter count, FLOPs, and memory footprint. It recommends deployment techniques (precision, quantization, flash attention) without modifying any graph structure. Use `OptimizationEngine` for graph-level transformations and `InferenceOptimizer` for deployment planning.

---

**Q: Does ChipBridge support quantization-aware training (QAT)?**

A: No. ChipBridge analyzes already-trained models for inference deployment. QAT is a training-time technique requiring framework-specific tooling (PyTorch's `torch.quantization`, TensorFlow's `tfmot`). ChipBridge's `InferenceOptimizer` recommends INT8 post-training quantization as a deployment technique, but the actual quantization must be performed in the appropriate framework before deployment.
