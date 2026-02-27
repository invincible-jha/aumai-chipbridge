"""Pydantic v2 models for cross-hardware ML inference optimization."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class HardwareTarget(str, Enum):
    """Supported hardware execution targets."""

    CPU_X86 = "cpu_x86"
    CPU_ARM = "cpu_arm"
    GPU_CUDA = "gpu_cuda"
    GPU_ROCM = "gpu_rocm"
    TPU = "tpu"
    NPU = "npu"
    A100 = "a100"
    T4 = "t4"
    M2 = "m2"
    XEON = "xeon"
    ARM = "arm"


class OperatorType(str, Enum):
    """Types of ML operators in a computation graph."""

    CONV2D = "conv2d"
    MATMUL = "matmul"
    ATTENTION = "attention"
    LAYERNORM = "layernorm"
    SOFTMAX = "softmax"
    GELU = "gelu"
    EMBEDDING = "embedding"


class GraphOperator(BaseModel):
    """A single operator node in a computation graph.

    Attributes:
        op_id: Unique identifier for this operator.
        op_type: The operator type.
        input_shapes: List of input tensor shapes.
        output_shape: Output tensor shape.
        attributes: Operator-specific attributes (e.g., kernel_size, stride).
    """

    op_id: str
    op_type: OperatorType
    input_shapes: list[tuple[int, ...]] = Field(default_factory=list)
    output_shape: tuple[int, ...] = Field(default_factory=tuple)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("op_id")
    @classmethod
    def validate_op_id(cls, value: str) -> str:
        """Ensure op_id is not blank."""
        if not value.strip():
            raise ValueError("op_id must not be blank.")
        return value.strip()


class ModelGraph(BaseModel):
    """A computation graph representing an ML model.

    Attributes:
        name: Human-readable name for the model.
        operators: List of operators in the graph.
        edges: Directed edges as (source_op_id, dest_op_id) pairs.
    """

    name: str
    operators: list[GraphOperator] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)

    def operator_by_id(self, op_id: str) -> Optional[GraphOperator]:
        """Look up an operator by its ID."""
        for op in self.operators:
            if op.op_id == op_id:
                return op
        return None


class OptimizationPass(BaseModel):
    """Describes a graph optimization transformation.

    Attributes:
        name: Short name of the optimization pass.
        description: Explanation of what the pass does.
        applicable_targets: Hardware targets where this pass is beneficial.
    """

    name: str
    description: str
    applicable_targets: list[HardwareTarget] = Field(default_factory=list)


class HardwareProfile(BaseModel):
    """Hardware capability profile.

    Attributes:
        target: The hardware target this profile describes.
        compute_tflops: Peak compute throughput in TFLOPS (FP32).
        memory_bandwidth_gbps: Peak memory bandwidth in GB/s.
        memory_capacity_gb: Total on-chip memory in GB.
        has_tensor_cores: Whether tensor / matrix cores are available.
        supports_fp16: FP16 hardware support.
        supports_int8: INT8 hardware support.
        architecture: Short architecture name string.
    """

    target: HardwareTarget
    compute_tflops: float = Field(gt=0.0)
    memory_bandwidth_gbps: float = Field(gt=0.0)
    memory_capacity_gb: float = Field(gt=0.0)
    has_tensor_cores: bool = Field(default=False)
    supports_fp16: bool = Field(default=True)
    supports_int8: bool = Field(default=True)
    architecture: str = Field(default="unknown")
    notes: str = Field(default="")


class ModelProfile(BaseModel):
    """Computational profile of an ML model to be deployed.

    Attributes:
        model_name: Human-readable model name.
        parameter_count: Total trainable parameters.
        flops_per_inference: FLOPs for a single forward pass.
        memory_footprint_mb: Memory required for weights + activations in MB.
        batch_size: Expected inference batch size.
        has_attention: Whether the model uses attention mechanisms.
        has_convolutions: Whether the model uses convolutional layers.
    """

    model_name: str
    parameter_count: int = Field(default=0, ge=0)
    flops_per_inference: float = Field(default=0.0, ge=0.0)
    memory_footprint_mb: float = Field(default=0.0, ge=0.0)
    batch_size: int = Field(default=1, gt=0)
    has_attention: bool = Field(default=False)
    has_convolutions: bool = Field(default=False)
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class OptimizationPlan(BaseModel):
    """A recommended optimization plan for deploying a model on a target.

    Attributes:
        model_name: Name of the model.
        target_id: ID of the hardware target.
        recommended_techniques: Prioritised optimization technique names.
        expected_speedup: Estimated latency speedup vs. FP32 baseline.
        expected_memory_reduction: Fraction of memory reduction (0 to 1).
        estimated_latency_ms: Predicted inference latency in milliseconds.
        estimated_throughput_qps: Predicted queries per second.
        precision_recommendation: Recommended operating precision.
        warnings: Any compatibility warnings.
        notes: Additional notes.
    """

    model_name: str
    target_id: str
    recommended_techniques: list[str] = Field(default_factory=list)
    expected_speedup: float = Field(default=1.0, ge=0.0)
    expected_memory_reduction: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_latency_ms: float = Field(default=0.0, ge=0.0)
    estimated_throughput_qps: float = Field(default=0.0, ge=0.0)
    precision_recommendation: str = Field(default="fp32")
    warnings: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class OperatorPrediction(BaseModel):
    """Predicted performance of a single operator on a hardware target.

    Attributes:
        op_id: The operator ID.
        target: The hardware target.
        latency_ms: Predicted execution latency in milliseconds.
        roofline_bound: Whether the operator is compute or memory bound.
        flops: Estimated FLOPs for this operator.
    """

    op_id: str
    target: HardwareTarget
    latency_ms: float = Field(ge=0.0)
    roofline_bound: str = Field(default="compute")
    flops: float = Field(default=0.0, ge=0.0)


class InferencePrediction(BaseModel):
    """Predicted end-to-end inference performance.

    Attributes:
        model_name: Name of the model.
        target: Hardware target.
        total_latency_ms: Sum of per-operator latency predictions.
        operator_predictions: Per-operator breakdown.
        bottleneck_op_id: ID of the slowest operator.
    """

    model_name: str
    target: HardwareTarget
    total_latency_ms: float = Field(ge=0.0)
    operator_predictions: list[OperatorPrediction] = Field(default_factory=list)
    bottleneck_op_id: Optional[str] = None


__all__ = [
    "HardwareTarget",
    "OperatorType",
    "GraphOperator",
    "ModelGraph",
    "OptimizationPass",
    "HardwareProfile",
    "ModelProfile",
    "OptimizationPlan",
    "OperatorPrediction",
    "InferencePrediction",
]
