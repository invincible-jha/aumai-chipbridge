# API Reference — aumai-chipbridge

Complete reference for all public classes, functions, and Pydantic models.

---

## Module: `aumai_chipbridge.models`

### `HardwareTarget`

```python
class HardwareTarget(str, Enum)
```

Enumeration of all supported hardware execution targets.

| Member      | Value        | Description                                  |
|-------------|--------------|----------------------------------------------|
| `CPU_X86`   | `"cpu_x86"`  | Generic x86-64 CPU                           |
| `CPU_ARM`   | `"cpu_arm"`  | Generic ARM CPU                              |
| `GPU_CUDA`  | `"gpu_cuda"` | Generic NVIDIA CUDA GPU                      |
| `GPU_ROCM`  | `"gpu_rocm"` | AMD ROCm GPU                                 |
| `TPU`       | `"tpu"`      | Google TPU                                   |
| `NPU`       | `"npu"`      | Generic Neural Processing Unit               |
| `A100`      | `"a100"`     | NVIDIA A100 80GB HBM2e (Ampere)              |
| `T4`        | `"t4"`       | NVIDIA T4 16GB GDDR6 (Turing)                |
| `M2`        | `"m2"`       | Apple M2 with unified memory (Apple Silicon) |
| `XEON`      | `"xeon"`     | Intel Xeon Scalable with AVX-512             |
| `ARM`       | `"arm"`      | ARM Cortex-A78 with NEON SIMD                |

Since `HardwareTarget` extends `str`, values can be used directly as strings or passed to `click.Choice`.

---

### `OperatorType`

```python
class OperatorType(str, Enum)
```

Types of ML computation operators that can appear in a model graph.

| Member       | Value          | Description                                |
|--------------|----------------|--------------------------------------------|
| `CONV2D`     | `"conv2d"`     | 2D convolution                             |
| `MATMUL`     | `"matmul"`     | Matrix multiply / linear layer             |
| `ATTENTION`  | `"attention"`  | Scaled dot-product attention               |
| `LAYERNORM`  | `"layernorm"`  | Layer normalization                        |
| `SOFTMAX`    | `"softmax"`    | Softmax activation                         |
| `GELU`       | `"gelu"`       | Gaussian error linear unit activation      |
| `EMBEDDING`  | `"embedding"`  | Token / lookup embedding                   |

---

### `GraphOperator`

```python
class GraphOperator(BaseModel)
```

A single operator node in a model computation graph.

**Fields:**

| Field           | Type                      | Required | Default | Description                                      |
|-----------------|---------------------------|----------|---------|--------------------------------------------------|
| `op_id`         | `str`                     | Yes      |         | Unique identifier for this operator. Must not be blank. |
| `op_type`       | `OperatorType`            | Yes      |         | The operator type.                               |
| `input_shapes`  | `list[tuple[int, ...]]`   | No       | `[]`    | List of input tensor shapes (one per input tensor). |
| `output_shape`  | `tuple[int, ...]`         | No       | `()`    | Output tensor shape.                             |
| `attributes`    | `dict[str, Any]`          | No       | `{}`    | Operator-specific parameters (e.g., `"kernel_h": 3`, `"num_heads": 12`, `"embed_dim": 768`). |

**Validators:**
- `op_id`: Strips whitespace, raises `ValueError` if blank after stripping.

**Attribute conventions by operator type:**

| Operator Type | Common Attributes                                             |
|---------------|---------------------------------------------------------------|
| `CONV2D`      | `out_channels` (int), `kernel_h` (int, default 3), `kernel_w` (int, default 3), `preferred_layout` (str, added by layout pass) |
| `ATTENTION`   | `num_heads` (int)                                            |
| `EMBEDDING`   | `embed_dim` (int, default 768), `vocab_size` (int)          |

**Example:**

```python
from aumai_chipbridge import GraphOperator, OperatorType

op = GraphOperator(
    op_id="attention_layer_0",
    op_type=OperatorType.ATTENTION,
    input_shapes=[(1, 128, 768)],
    output_shape=(1, 128, 768),
    attributes={"num_heads": 12},
)
```

---

### `ModelGraph`

```python
class ModelGraph(BaseModel)
```

A computation graph representing an ML model as a directed acyclic graph (DAG) of operators.

**Fields:**

| Field       | Type                        | Required | Default | Description                                        |
|-------------|-----------------------------|----------|---------|----------------------------------------------------|
| `name`      | `str`                       | Yes      |         | Human-readable model name.                         |
| `operators` | `list[GraphOperator]`       | No       | `[]`    | All operator nodes in the graph.                   |
| `edges`     | `list[tuple[str, str]]`     | No       | `[]`    | Directed edges as `(source_op_id, dest_op_id)` pairs. |

**Methods:**

#### `operator_by_id(op_id: str) -> Optional[GraphOperator]`

Look up an operator by its ID. Returns `None` if not found.

```python
from aumai_chipbridge import ModelGraph, GraphOperator, OperatorType

graph = ModelGraph(
    name="my_model",
    operators=[
        GraphOperator(op_id="ff", op_type=OperatorType.MATMUL,
                      input_shapes=[(128, 768)], output_shape=(128, 3072)),
    ],
    edges=[],
)
op = graph.operator_by_id("ff")  # returns the MATMUL operator
missing = graph.operator_by_id("nonexistent")  # returns None
```

**Serialization:**

```python
json_str = graph.model_dump_json(indent=2)
graph2 = ModelGraph.model_validate_json(json_str)
```

---

### `HardwareProfile`

```python
class HardwareProfile(BaseModel)
```

Hardware capability profile describing peak compute and memory specifications.

**Fields:**

| Field                    | Type            | Required | Constraints | Description                                     |
|--------------------------|-----------------|----------|-------------|--------------------------------------------------|
| `target`                 | `HardwareTarget`| Yes      |             | The hardware target this profile describes.      |
| `compute_tflops`         | `float`         | Yes      | `> 0.0`     | Peak compute throughput in TFLOPS (FP32).        |
| `memory_bandwidth_gbps`  | `float`         | Yes      | `> 0.0`     | Peak memory bandwidth in GB/s.                  |
| `memory_capacity_gb`     | `float`         | Yes      | `> 0.0`     | Total on-chip memory in GB.                     |
| `has_tensor_cores`       | `bool`          | No       | `False`     | Whether tensor/matrix cores are available.       |
| `supports_fp16`          | `bool`          | No       | `True`      | FP16 hardware support.                          |
| `supports_int8`          | `bool`          | No       | `True`      | INT8 hardware support.                          |
| `architecture`           | `str`           | No       | `"unknown"` | Short architecture name string.                 |
| `notes`                  | `str`           | No       | `""`        | Optional human-readable notes.                  |

```python
from aumai_chipbridge.models import HardwareProfile, HardwareTarget

profile = HardwareProfile(
    target=HardwareTarget.A100,
    compute_tflops=77.6,
    memory_bandwidth_gbps=2000.0,
    memory_capacity_gb=80.0,
    has_tensor_cores=True,
    supports_fp16=True,
    supports_int8=True,
    architecture="ampere",
    notes="NVIDIA A100 80GB HBM2e",
)
```

---

### `ModelProfile`

```python
class ModelProfile(BaseModel)
```

High-level computational profile of an ML model for deployment planning.

**Fields:**

| Field                   | Type              | Required | Constraints | Description                                         |
|-------------------------|-------------------|----------|-------------|-----------------------------------------------------|
| `model_name`            | `str`             | Yes      |             | Human-readable model name.                          |
| `parameter_count`       | `int`             | No       | `>= 0`      | Total trainable parameters.                         |
| `flops_per_inference`   | `float`           | No       | `>= 0.0`    | FLOPs for a single forward pass.                    |
| `memory_footprint_mb`   | `float`           | No       | `>= 0.0`    | Memory required for weights + activations in MB.    |
| `batch_size`            | `int`             | No       | `> 0`       | Expected inference batch size. Default: `1`.        |
| `has_attention`         | `bool`            | No       | `False`     | Whether the model uses attention mechanisms.         |
| `has_convolutions`      | `bool`            | No       | `False`     | Whether the model uses convolutional layers.         |
| `extra_metadata`        | `dict[str, Any]`  | No       | `{}`        | Arbitrary additional metadata.                      |

---

### `OptimizationPass`

```python
class OptimizationPass(BaseModel)
```

Describes a graph optimization transformation.

**Fields:**

| Field                  | Type                    | Description                                              |
|------------------------|-------------------------|----------------------------------------------------------|
| `name`                 | `str`                   | Short identifier for the pass.                          |
| `description`          | `str`                   | Human-readable explanation of what the pass does.        |
| `applicable_targets`   | `list[HardwareTarget]`  | Hardware targets where this pass is beneficial.          |

---

### `OptimizationPlan`

```python
class OptimizationPlan(BaseModel)
```

A recommended optimization plan for deploying a model on a hardware target. Returned by `InferenceOptimizer.analyze()`.

**Fields:**

| Field                       | Type             | Constraints      | Description                                              |
|-----------------------------|------------------|------------------|----------------------------------------------------------|
| `model_name`                | `str`            |                  | Name of the model.                                       |
| `target_id`                 | `str`            |                  | ID of the hardware target.                              |
| `recommended_techniques`    | `list[str]`      |                  | Ordered list of optimization technique names.            |
| `expected_speedup`          | `float`          | `>= 0.0`         | Estimated latency speedup vs. FP32 baseline.            |
| `expected_memory_reduction` | `float`          | `>= 0.0, <= 1.0` | Fraction of memory reduction (0 to 1).                  |
| `estimated_latency_ms`      | `float`          | `>= 0.0`         | Predicted inference latency in milliseconds.            |
| `estimated_throughput_qps`  | `float`          | `>= 0.0`         | Predicted queries per second.                           |
| `precision_recommendation`  | `str`            |                  | Recommended operating precision: `"fp32"`, `"fp16"`. |
| `warnings`                  | `list[str]`      |                  | Compatibility warnings (e.g., model too large for device). |
| `notes`                     | `Optional[str]`  |                  | Additional notes about the prediction basis.            |

---

### `OperatorPrediction`

```python
class OperatorPrediction(BaseModel)
```

Predicted performance of a single operator on a hardware target.

**Fields:**

| Field            | Type            | Constraints  | Description                                              |
|------------------|-----------------|--------------|----------------------------------------------------------|
| `op_id`          | `str`           |              | The operator ID.                                         |
| `target`         | `HardwareTarget`|              | The hardware target.                                     |
| `latency_ms`     | `float`         | `>= 0.0`     | Predicted execution latency in milliseconds.             |
| `roofline_bound` | `str`           |              | Either `"compute"` or `"memory"`.                        |
| `flops`          | `float`         | `>= 0.0`     | Estimated FLOPs for this operator.                       |

---

### `InferencePrediction`

```python
class InferencePrediction(BaseModel)
```

Predicted end-to-end inference performance. Returned by `PerformancePredictor.predict()`.

**Fields:**

| Field                   | Type                        | Constraints | Description                                       |
|-------------------------|-----------------------------|-------------|---------------------------------------------------|
| `model_name`            | `str`                       |             | Name of the model.                                |
| `target`                | `HardwareTarget`            |             | Hardware target.                                  |
| `total_latency_ms`      | `float`                     | `>= 0.0`    | Sum of all per-operator latency predictions.      |
| `operator_predictions`  | `list[OperatorPrediction]`  |             | Per-operator breakdown.                           |
| `bottleneck_op_id`      | `Optional[str]`             |             | ID of the operator with the highest predicted latency. |

---

## Module: `aumai_chipbridge.core`

### `GraphAnalyzer`

```python
class GraphAnalyzer
```

Analyzes a `ModelGraph` for performance characteristics.

#### `topological_sort(graph: ModelGraph) -> list[str]`

Return operator IDs in topological execution order using Kahn's algorithm.

- **Returns**: `list[str]` — operator IDs in valid execution order (root operators first)
- **Raises**: `ValueError` if the graph contains a cycle

```python
from aumai_chipbridge import GraphAnalyzer

analyzer = GraphAnalyzer()
order = analyzer.topological_sort(graph)
```

#### `compute_flops(graph: ModelGraph) -> dict[str, float]`

Compute estimated FLOPs for each operator.

- **Returns**: `dict[str, float]` mapping `op_id` to estimated FLOPs

#### `identify_bottleneck(graph: ModelGraph, target: HardwareTarget) -> Optional[str]`

Identify the operator with the highest predicted latency for the given target.

- **Returns**: The `op_id` of the slowest operator, or `None` if the graph has no operators

```python
bottleneck = analyzer.identify_bottleneck(graph, HardwareTarget.A100)
```

#### `total_flops(graph: ModelGraph) -> float`

Sum FLOPs across all operators.

- **Returns**: `float` — total estimated FLOPs

---

### `HardwareMapper`

```python
class HardwareMapper
```

Maps operators to hardware-native implementations and estimates per-operator latency.

#### `get_implementation(op: GraphOperator, target: HardwareTarget) -> str`

Return the recommended implementation name for an operator on a given target.

- **Returns**: `str` — e.g., `"cublas_tensor_core_gemm"`, `"flash_attention_v2_fp16"`, `"mkl_avx512_sgemm"`

Implementation mappings (selected examples):

| Operator    | Target    | Implementation                  |
|-------------|-----------|---------------------------------|
| MATMUL      | A100      | `cublas_tensor_core_gemm`       |
| MATMUL      | XEON      | `mkl_avx512_sgemm`              |
| MATMUL      | M2        | `accelerate_blas_gemm`          |
| CONV2D      | A100      | `cudnn_amp_conv2d`              |
| ATTENTION   | A100      | `flash_attention_v2_fp16`       |
| ATTENTION   | TPU       | `tpu_attention`                 |

```python
from aumai_chipbridge.core import HardwareMapper
from aumai_chipbridge.models import GraphOperator, OperatorType, HardwareTarget

mapper = HardwareMapper()
op = GraphOperator(op_id="ff", op_type=OperatorType.MATMUL,
                   input_shapes=[(128, 768)], output_shape=(128, 3072))
impl = mapper.get_implementation(op, HardwareTarget.A100)
# Returns: "cublas_tensor_core_gemm"
```

#### `estimate_operator_latency(op: GraphOperator, target: HardwareTarget) -> float`

Estimate operator execution time in milliseconds using the roofline model plus hardware-specific speedup multipliers.

- **Returns**: `float` — estimated latency in milliseconds (minimum `0.001`)

---

### `OptimizationEngine`

```python
class OptimizationEngine
```

Applies graph optimization passes to a `ModelGraph`.

#### `list_passes(target: Optional[HardwareTarget] = None) -> list[OptimizationPass]`

List available optimization passes, optionally filtered by hardware target.

- **Parameters**: `target` — if provided, only passes applicable to this target are returned; if `None`, all passes are returned
- **Returns**: `list[OptimizationPass]`

Available passes:

| Pass Name                  | Applicable Targets                                   |
|----------------------------|------------------------------------------------------|
| `operator_fusion`          | All targets                                          |
| `constant_folding`         | All targets                                          |
| `dead_code_elimination`    | All targets                                          |
| `layout_optimization`      | GPU_CUDA, GPU_ROCM, A100, T4, TPU only               |

#### `optimize(graph: ModelGraph, target: HardwareTarget, passes: Optional[list[str]] = None) -> tuple[ModelGraph, list[str]]`

Apply optimization passes to the graph.

- **Parameters**:
  - `graph` — the source `ModelGraph`
  - `target` — the `HardwareTarget` to optimize for
  - `passes` — optional list of pass names; if `None`, all applicable passes are applied
- **Returns**: `tuple[ModelGraph, list[str]]` — optimized graph and list of applied pass names

```python
from aumai_chipbridge import OptimizationEngine, HardwareTarget

engine = OptimizationEngine()
optimized, applied = engine.optimize(graph, HardwareTarget.A100)
# Or with specific passes:
optimized, applied = engine.optimize(
    graph, HardwareTarget.A100, passes=["operator_fusion", "layout_optimization"]
)
```

---

### `PerformancePredictor`

```python
class PerformancePredictor
```

Predicts inference latency using the roofline performance model.

#### `predict(graph: ModelGraph, target: HardwareTarget) -> InferencePrediction`

Predict end-to-end inference latency for a model on a hardware target.

- **Returns**: `InferencePrediction` with total latency, per-operator breakdown, and bottleneck identification

```python
from aumai_chipbridge import PerformancePredictor, HardwareTarget

predictor = PerformancePredictor()
pred = predictor.predict(graph, HardwareTarget.A100)

print(f"Total: {pred.total_latency_ms:.4f} ms")
print(f"Bottleneck: {pred.bottleneck_op_id}")
for op_pred in sorted(pred.operator_predictions, key=lambda p: -p.latency_ms):
    print(f"  {op_pred.op_id}: {op_pred.latency_ms:.6f} ms [{op_pred.roofline_bound}]")
```

---

### `CrossHardwareComparator`

```python
class CrossHardwareComparator
```

Compares model inference performance across multiple hardware targets.

#### `compare(graph: ModelGraph, targets: Optional[list[HardwareTarget]] = None) -> dict[HardwareTarget, InferencePrediction]`

Compare model performance across hardware targets. If `targets` is `None`, all registered targets are compared.

#### `best_target(graph: ModelGraph, targets: Optional[list[HardwareTarget]] = None) -> HardwareTarget`

Identify the hardware target with the lowest predicted total latency.

#### `speedup_table(graph: ModelGraph, baseline: HardwareTarget = HardwareTarget.CPU_X86) -> dict[str, float]`

Compute speedup of each target relative to a baseline.

- **Returns**: `dict[str, float]` mapping target value strings to speedup multipliers

```python
from aumai_chipbridge import CrossHardwareComparator, HardwareTarget

comparator = CrossHardwareComparator()
best = comparator.best_target(graph)
speedups = comparator.speedup_table(graph, baseline=HardwareTarget.CPU_X86)
for target_id, speedup in sorted(speedups.items(), key=lambda x: -x[1]):
    print(f"  {target_id}: {speedup:.1f}x")
```

---

### `HardwareRegistry`

```python
class HardwareRegistry
```

Registry of `HardwareProfile` instances. Pre-loaded with profiles for all 11 built-in targets.

#### `get(target_id: str) -> HardwareProfile`

Retrieve a hardware profile by target ID string (case-insensitive).

- **Raises**: `KeyError` with a message listing all available IDs if the target is not found

```python
from aumai_chipbridge.core import HardwareRegistry

registry = HardwareRegistry()
profile = registry.get("a100")
print(profile.compute_tflops)  # 77.6
```

#### `list_targets() -> list[str]`

Return all registered target ID strings in sorted order.

- **Returns**: e.g., `['a100', 'arm', 'cpu_arm', 'cpu_x86', 'gpu_cuda', 'gpu_rocm', 'm2', 'npu', 't4', 'tpu', 'xeon']`

#### `register(profile: HardwareProfile) -> None`

Register a custom hardware profile, overwriting any existing profile for the same target.

```python
from aumai_chipbridge.models import HardwareProfile, HardwareTarget

registry.register(HardwareProfile(
    target=HardwareTarget.NPU,
    compute_tflops=12.0,
    memory_bandwidth_gbps=400.0,
    memory_capacity_gb=16.0,
    has_tensor_cores=True,
    supports_fp16=True,
    supports_int8=True,
    architecture="custom_npu_v4",
))
```

---

### `InferenceOptimizer`

```python
class InferenceOptimizer
```

Recommends deployment optimization plans for a model on a hardware target.

#### `__init__(self, registry: Optional[HardwareRegistry] = None) -> None`

#### `analyze(model: ModelProfile, target_id: str) -> OptimizationPlan`

Generate an optimization plan for the model on the target hardware.

- **Raises**: `KeyError` if the target ID is not found in the registry

**Technique selection logic:**

| Condition                                                     | Technique Added           |
|---------------------------------------------------------------|---------------------------|
| `hw.supports_fp16 and hw.has_tensor_cores`                   | `fp16_mixed_precision`    |
| `hw.supports_fp16` (without tensor cores)                    | `fp16_mixed_precision`    |
| `model.has_attention` and target is CUDA/ROCm GPU and fp16   | `flash_attention`         |
| `hw.supports_int8`                                           | `int8_quantization`       |
| Model footprint > 70% of device memory                       | Warning added to `warnings` |
| Always                                                        | `kernel_fusion`, `operator_fusion` |

```python
from aumai_chipbridge.core import InferenceOptimizer
from aumai_chipbridge.models import ModelProfile

optimizer = InferenceOptimizer()
plan = optimizer.analyze(
    ModelProfile(
        model_name="llama-7b",
        parameter_count=7_000_000_000,
        flops_per_inference=14e12,
        memory_footprint_mb=14_000,
        has_attention=True,
    ),
    "a100",
)
print(plan.precision_recommendation)  # "fp16"
print(plan.expected_speedup)          # 2.0
```

---

### `CrossCompiler`

```python
class CrossCompiler
```

Generates deployment artefact descriptions for cross-hardware compilation.

#### `__init__(self, registry: Optional[HardwareRegistry] = None) -> None`

#### `compile(model: ModelProfile, target_id: str) -> dict[str, object]`

Generate a deployment artefact description.

- **Raises**: `KeyError` if the target ID is not found

Return dictionary keys:

| Key                    | Type           | Description                                               |
|------------------------|----------------|-----------------------------------------------------------|
| `model_name`           | `str`          | Model name.                                               |
| `target`               | `str`          | Target enum value string.                                 |
| `architecture`         | `str`          | Architecture string from `HardwareProfile`.               |
| `export_command`       | `str`          | Framework-specific export/compile command.                |
| `compile_flags`        | `list[str]`    | Compiler flags for the target.                            |
| `runtime_requirements` | `list[str]`    | Required runtime library versions.                        |
| `memory_required_gb`   | `float`        | Model memory footprint in GB.                            |
| `device_memory_gb`     | `float`        | Device total memory capacity in GB.                      |
| `fits_in_device_memory`| `bool`         | Whether the model fits in device memory.                 |
| `precision_support`    | `dict`         | `{"fp32": bool, "fp16": bool, "int8": bool, "tensor_cores": bool}` |

---

## Module-Level Constants

### `HARDWARE_PROFILES`

```python
# aumai_chipbridge.core
HARDWARE_PROFILES: dict[HardwareTarget, HardwareProfile]
```

Module-level dictionary mapping every `HardwareTarget` to its built-in `HardwareProfile`. Used internally by all core classes. Do not mutate directly; use `HardwareRegistry.register()` instead.

```python
from aumai_chipbridge.core import HARDWARE_PROFILES, HardwareTarget

a100 = HARDWARE_PROFILES[HardwareTarget.A100]
print(a100.compute_tflops)        # 77.6
print(a100.memory_bandwidth_gbps) # 2000.0
print(a100.architecture)          # "ampere"
```

---

## Public Re-exports from `aumai_chipbridge`

```python
from aumai_chipbridge import (
    HardwareTarget,
    OperatorType,
    GraphOperator,
    ModelGraph,
    OptimizationPass,
    HardwareProfile,
    OperatorPrediction,
    InferencePrediction,
    GraphAnalyzer,
    HardwareMapper,
    OptimizationEngine,
    PerformancePredictor,
    CrossHardwareComparator,
)
```

`ModelProfile`, `OptimizationPlan`, `HardwareRegistry`, `InferenceOptimizer`, and `CrossCompiler` are available from `aumai_chipbridge.models` and `aumai_chipbridge.core`.

---

## Error Reference

| Exception      | Raised By                                | Condition                                             |
|----------------|------------------------------------------|-------------------------------------------------------|
| `ValueError`   | `GraphOperator` validator                | `op_id` is blank after stripping whitespace            |
| `ValueError`   | `GraphAnalyzer.topological_sort`         | Graph contains a cycle                                |
| `KeyError`     | `HardwareRegistry.get`                   | No profile found for the given target ID             |
| `KeyError`     | `InferenceOptimizer.analyze`             | Propagated from `HardwareRegistry.get`                |
| `KeyError`     | `CrossCompiler.compile`                  | Propagated from `HardwareRegistry.get`                |
| `SystemExit`   | CLI `_load_graph`                        | JSON file not found or fails Pydantic validation      |
