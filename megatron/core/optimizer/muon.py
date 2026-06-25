# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron muon optimizer wrapper to handle tensor-parallel."""

import math
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_size, log_single_rank

from . import HAVE_EMERGING_OPTIMIZERS, _get_param_groups, get_megatron_optimizer
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey

if HAVE_EMERGING_OPTIMIZERS:
    from emerging_optimizers import utils
    from emerging_optimizers.orthogonalized_optimizers import (
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp
else:
    OrthogonalizedOptimizer = object

if HAVE_EMERGING_OPTIMIZERS:
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT


logger = logging.getLogger(__name__)


def get_supported_coefficient_types() -> tuple[str, ...]:
    """Return the coefficient types supported by the installed emerging_optimizers.

    Reads the members of the ``NSCoeffT`` Literal type so that new types
    added upstream are automatically available without code changes here.
    """
    assert (
        HAVE_EMERGING_OPTIMIZERS
    ), "emerging_optimizers >= 0.2 is required for NSCoeffT. Please install or upgrade it."
    return get_args(NSCoeffT)  # pylint: disable=possibly-used-before-assignment


def validate_coefficient_type(coefficient_type: str) -> None:
    """Raise ``ValueError`` if *coefficient_type* is not supported."""
    supported = get_supported_coefficient_types()
    if coefficient_type not in supported:
        raise ValueError(
            f"Unsupported muon coefficient type '{coefficient_type}'. "
            f"Supported types: {supported}"
        )


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        kwargs: Dict[str, Any] = {},
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        validate_coefficient_type(coefficient_type)

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            mode_value = "duplicated" if mode == "blockwise" else mode
            mode_kwarg = {"tp_mode": mode_value}
            ns_kwargs = dict(
                steps=num_ns_steps, tp_group=tp_group, partition_dim=partition_dim, **mode_kwarg
            )
            ns_kwargs["coefficient_type"] = coefficient_type
            # pylint: disable-next=possibly-used-before-assignment
            orth_grad = newton_schulz_tp(grad, **ns_kwargs)
            # pylint: disable-next=possibly-used-before-assignment
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.mode = mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        nesterov_kwarg = {"nesterov": use_nesterov}
        super().__init__(
            params,
            lr,
            momentum_beta,
            **nesterov_kwarg,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to momentum
                because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            # emerging-optimizers use None instead of -1 to indicate no tensor parallel
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            # split grouped attention parameters (e.g., QKV, GQA, etc.)
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, split shapes {self.qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),
                self.qkv_split_shapes,
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            # Apply Newton-Schulz and scales to each component, concat back
            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad


class AdaptiveTensorParallelMuon(OrthogonalizedOptimizer):
    """
    SNECV-Muon:
      - Energy-only trigger
      - Self-normalized Energy-CV score
      - Leaky medium-band pressure accumulator
      - Optional communication budget rho via adaptive full-step threshold

    The trigger logic is:
      raw_cv  = shard-energy coefficient of variation
      mean_t  = EWMA mean of raw_cv
      var_t   = EWMA variance proxy of raw_cv
      z_t     = (raw_cv - mean_t) / sqrt(var_t + eps)

    Actions:
      z_t >= tau_high -> full orth, lr_full
      z_low <= z_t < tau_high -> build leaky pressure and decay local lr_block
      pressure_t >= H -> full orth, lr_full
      otherwise -> local blockwise orth, lr_block

    The pressure state is
      C_t = gamma * C_{t-1}
            + 1[z_t in medium] * (1 + alpha * (z_t - z_low) / (tau_high - z_low + eps))

    If comm_budget_rho is not None, tau_high is adapted online so that the long-run
    fraction of full steps tracks rho.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional["ProcessGroupCollection"] = None,
        mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        lr_block: float | None = None,
        lr_full: float | None = None,
        # SNECV-Muon knobs
        snecv_beta: float = 0.95,
        snecv_eps: float = 1e-6,
        snecv_z_low: float = 1.0,
        snecv_z_high: float = 3.0,
        snecv_warmup_steps: int = 100,
        snecv_pressure_gamma: float = 0.95,
        snecv_pressure_alpha: float = 1.0,
        snecv_pressure_threshold_h: float = 4.0,
        snecv_pressure_reset_factor: float = 0.0,
        snecv_local_lr_gamma: float = 0.5,
        snecv_use_smooth_local_lr_decay: bool = False,
        snecv_monitor_signal: Literal[
            "energy_cv",
            "stable_rank_cv",
            "spectral_norm_cv",
            "directional_gram_cv",
            "gram_sketch_distance",
            "subspace_angle_sketch",
        ] = "energy_cv",
        snecv_monitor_sketch_q: int = 4,
        snecv_monitor_power_iters: int = 2,
        # optional communication budget for full orth ratio
        comm_budget_rho: float | None = None,
        snecv_stats_log_interval: int = 0,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if not (0.0 <= snecv_beta < 1.0):
            raise ValueError(f"snecv_beta must be in [0, 1), got {snecv_beta}")
        if snecv_warmup_steps < 0:
            raise ValueError(f"snecv_warmup_steps must be >= 0, got {snecv_warmup_steps}")
        if snecv_z_low < 0.0:
            raise ValueError(f"snecv_z_low must be >= 0, got {snecv_z_low}")
        if snecv_z_high <= snecv_z_low:
            raise ValueError(
                f"snecv_z_high must be > snecv_z_low, got {snecv_z_high} <= {snecv_z_low}"
            )
        if not (0.0 <= snecv_pressure_gamma <= 1.0):
            raise ValueError(
                f"snecv_pressure_gamma must be in [0, 1], got {snecv_pressure_gamma}"
            )
        if snecv_pressure_alpha < 0.0:
            raise ValueError(
                f"snecv_pressure_alpha must be >= 0, got {snecv_pressure_alpha}"
            )
        if snecv_pressure_threshold_h <= 0.0:
            raise ValueError(
                f"snecv_pressure_threshold_h must be > 0, got {snecv_pressure_threshold_h}"
            )
        if not (0.0 <= snecv_pressure_reset_factor <= 1.0):
            raise ValueError(
                "snecv_pressure_reset_factor must be in [0, 1], "
                f"got {snecv_pressure_reset_factor}"
            )
        if snecv_local_lr_gamma < 0.0:
            raise ValueError(
                f"snecv_local_lr_gamma must be >= 0, got {snecv_local_lr_gamma}"
            )
        valid_monitor_signals = {
            "energy_cv",
            "stable_rank_cv",
            "spectral_norm_cv",
            "directional_gram_cv",
            "gram_sketch_distance",
            "subspace_angle_sketch",
        }
        if snecv_monitor_signal not in valid_monitor_signals:
            raise ValueError(
                f"snecv_monitor_signal must be one of {sorted(valid_monitor_signals)}, "
                f"got {snecv_monitor_signal!r}"
            )
        if snecv_monitor_sketch_q < 1:
            raise ValueError(
                f"snecv_monitor_sketch_q must be >= 1, got {snecv_monitor_sketch_q}"
            )
        if snecv_monitor_power_iters < 0:
            raise ValueError(
                "snecv_monitor_power_iters must be >= 0, "
                f"got {snecv_monitor_power_iters}"
            )
        if comm_budget_rho is not None and not (0.0 < comm_budget_rho < 1.0):
            raise ValueError(
                f"comm_budget_rho must be in (0, 1) when provided, got {comm_budget_rho}"
            )
        if snecv_stats_log_interval < 0:
            raise ValueError(
                f"snecv_stats_log_interval must be >= 0, got {snecv_stats_log_interval}"
            )

        self.pg_collection = pg_collection
        self.mode = mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

        self.fp32_matmul_prec = fp32_matmul_prec
        self.coefficient_type = coefficient_type
        self.num_ns_steps = num_ns_steps
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor

        # SNECV state
        self.snecv_beta = snecv_beta
        self.snecv_eps = snecv_eps
        self.snecv_z_low = snecv_z_low
        self.snecv_z_high = snecv_z_high
        self.snecv_warmup_steps = snecv_warmup_steps
        self.snecv_pressure_gamma = snecv_pressure_gamma
        self.snecv_pressure_alpha = snecv_pressure_alpha
        self.snecv_pressure_threshold_h = snecv_pressure_threshold_h
        self.snecv_pressure_reset_factor = snecv_pressure_reset_factor
        self.snecv_local_lr_gamma = snecv_local_lr_gamma
        self.snecv_use_smooth_local_lr_decay = snecv_use_smooth_local_lr_decay
        self.snecv_monitor_signal = snecv_monitor_signal
        self.snecv_monitor_sketch_q = snecv_monitor_sketch_q
        self.snecv_monitor_power_iters = snecv_monitor_power_iters
        self.snecv_stats_log_interval = snecv_stats_log_interval

        # communication budget rho
        self.comm_budget_rho = comm_budget_rho
        self._budget_threshold = float(snecv_z_high)
        self._budget_updates = 0

        # step counter
        self._muon_global_step = 0

        # keep non-tensor metadata out of self.state[p]
        self._meta: Dict[int, Dict[str, Any]] = {}
        self._snecv_total_counts = self._new_snecv_counter_dict()
        self._snecv_step_counts = self._new_snecv_counter_dict()

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"

        if mode != "blockwise":
            raise ValueError(
                f"{self.__class__.__name__} only supports mode='blockwise', got {mode!r}. "
                "Use TensorParallelMuon for 'duplicated' or 'distributed'."
            )

        super().__init__(
            params,
            lr,
            momentum_beta,
            nesterov=use_nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=None,  # handled in orthogonalize()
        )

    def _param_id(self, p: torch.Tensor) -> int:
        return id(p)

    @staticmethod
    def _new_snecv_counter_dict() -> Dict[str, int]:
        return {
            "full_muon_ops": 0,
            "local_muon_ops": 0,
            "full_due_to_z": 0,
            "full_due_to_pressure": 0,
            "z_above_high": 0,
            "z_between": 0,
            "z_below_low": 0,
        }

    def _get_meta(self, p: torch.Tensor) -> Dict[str, Any]:
        pid = self._param_id(p)
        if pid not in self._meta:
            self._meta[pid] = {}
        return self._meta[pid]

    def _record_snecv_decision(
        self,
        z: float,
        threshold: float,
        use_full: bool,
        full_reason: str | None = None,
    ) -> None:
        if z >= threshold:
            band_key = "z_above_high"
        elif z >= self.snecv_z_low:
            band_key = "z_between"
        else:
            band_key = "z_below_low"

        op_key = "full_muon_ops" if use_full else "local_muon_ops"
        for counters in (self._snecv_total_counts, self._snecv_step_counts):
            counters[band_key] += 1
            counters[op_key] += 1
            if use_full and full_reason == "z":
                counters["full_due_to_z"] += 1
            elif use_full and full_reason == "pressure":
                counters["full_due_to_pressure"] += 1

    def get_snecv_frequency_stats(self, reset_step: bool = False) -> Dict[str, Dict[str, int]]:
        stats = {
            "total": dict(self._snecv_total_counts),
            "step": dict(self._snecv_step_counts),
        }
        if reset_step:
            self._snecv_step_counts = self._new_snecv_counter_dict()
        return stats

    def _log_snecv_frequency_stats(self) -> None:
        if self.snecv_stats_log_interval <= 0:
            return
        if self._muon_global_step % self.snecv_stats_log_interval != 0:
            return

        stats = self.get_snecv_frequency_stats(reset_step=False)
        step_stats = stats["step"]
        total_stats = stats["total"]
        log_single_rank(
            logger,
            logging.INFO,
            "[SNECV-Muon][stats] "
            f"step={self._muon_global_step} "
            f"step_full={step_stats['full_muon_ops']} "
            f"step_local={step_stats['local_muon_ops']} "
            f"step_full_due_to_z={step_stats['full_due_to_z']} "
            f"step_full_due_to_pressure={step_stats['full_due_to_pressure']} "
            f"step_z_above_high={step_stats['z_above_high']} "
            f"step_z_between={step_stats['z_between']} "
            f"step_z_below_low={step_stats['z_below_low']} "
            f"total_full={total_stats['full_muon_ops']} "
            f"total_local={total_stats['local_muon_ops']} "
            f"total_full_due_to_z={total_stats['full_due_to_z']} "
            f"total_full_due_to_pressure={total_stats['full_due_to_pressure']} "
            f"total_z_above_high={total_stats['z_above_high']} "
            f"total_z_between={total_stats['z_between']} "
            f"total_z_below_low={total_stats['z_below_low']}",
        )

    @staticmethod
    def _all_reduce_scalar(x: torch.Tensor, group: torch.distributed.ProcessGroup) -> torch.Tensor:
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM, group=group)
        return x

    def _cross_rank_cv(
        self,
        value: torch.Tensor,
        tp_group: torch.distributed.ProcessGroup,
    ) -> float:
        """Coefficient of variation of a scalar monitor value across TP ranks."""
        eps = 1e-12
        value = value.float()
        value2 = value * value
        value_sum = value.detach().clone()
        value2_sum = value2.detach().clone()
        self._all_reduce_scalar(value_sum, tp_group)
        self._all_reduce_scalar(value2_sum, tp_group)

        world = float(tp_group.size())
        mean = value_sum / world
        var = (value2_sum / world) - mean * mean
        var = torch.clamp(var, min=0.0)
        cv = (torch.sqrt(var) / (mean.abs() + eps)).item()
        return float(cv)

    def _energy_cv(self, M: torch.Tensor, tp_group: torch.distributed.ProcessGroup) -> float:
        """Instantaneous shard-energy coefficient of variation across TP ranks."""
        M_float = M.float()
        return self._cross_rank_cv((M_float * M_float).sum(), tp_group)

    def _spectral_norm_estimate(self, M: torch.Tensor) -> torch.Tensor:
        M_float = M.float()
        n = M_float.size(-1)
        v = torch.ones(n, device=M.device, dtype=torch.float32)
        v = v / (v.norm() + 1e-12)
        for _ in range(max(1, self.snecv_monitor_power_iters)):
            u = M_float.matmul(v)
            u = u / (u.norm() + 1e-12)
            v = M_float.t().matmul(u)
            v = v / (v.norm() + 1e-12)
        return M_float.matmul(v).norm()

    def _stable_rank_cv(self, M: torch.Tensor, tp_group: torch.distributed.ProcessGroup) -> float:
        M_float = M.float()
        fro_sq = (M_float * M_float).sum()
        spectral_sq = self._spectral_norm_estimate(M).pow(2).clamp_min(1e-12)
        return self._cross_rank_cv(fro_sq / spectral_sq, tp_group)

    def _spectral_norm_cv(self, M: torch.Tensor, tp_group: torch.distributed.ProcessGroup) -> float:
        return self._cross_rank_cv(self._spectral_norm_estimate(M), tp_group)

    def _deterministic_probe_matrix(
        self,
        num_cols: int,
        q: int,
        device: torch.device,
    ) -> torch.Tensor:
        row_idx = torch.arange(num_cols, device=device, dtype=torch.float32).unsqueeze(1)
        col_idx = torch.arange(q, device=device, dtype=torch.float32).unsqueeze(0)
        probes = torch.sin((row_idx + 1.0) * (col_idx + 1.0) * 12.9898)
        probes = probes + torch.cos((row_idx + 1.0) * (col_idx + 1.0) * 78.233)
        return probes / (probes.norm(dim=0, keepdim=True) + 1e-12)

    def _directional_gram_cv(self, M: torch.Tensor, tp_group: torch.distributed.ProcessGroup) -> float:
        M_float = M.float()
        probes = self._deterministic_probe_matrix(
            M_float.size(-1),
            self.snecv_monitor_sketch_q,
            M.device,
        )
        projected = M_float.matmul(probes)
        directional_energy = (projected * projected).sum(dim=0)
        cvs = [self._cross_rank_cv(v, tp_group) for v in directional_energy]
        return float(sum(cvs) / len(cvs))

    def _gram_sketch_distance(
        self,
        M: torch.Tensor,
        tp_group: torch.distributed.ProcessGroup,
    ) -> float:
        M_float = M.float()
        probes = self._deterministic_probe_matrix(
            M_float.size(-1),
            self.snecv_monitor_sketch_q,
            M.device,
        )
        sketch = M_float.matmul(probes)
        gram_sketch = sketch.t().matmul(sketch).flatten()
        mean_sketch = gram_sketch.detach().clone()
        self._all_reduce_scalar(mean_sketch, tp_group)
        mean_sketch = mean_sketch / float(tp_group.size())
        dist = (gram_sketch - mean_sketch).norm()
        mean_norm = mean_sketch.norm().clamp_min(1e-12)
        return float((dist / mean_norm).item())

    def _subspace_angle_sketch(
        self,
        M: torch.Tensor,
        tp_group: torch.distributed.ProcessGroup,
    ) -> float:
        M_float = M.float()
        q = min(self.snecv_monitor_sketch_q, M_float.size(-1), M_float.size(-2))
        probes = self._deterministic_probe_matrix(M_float.size(-1), q, M.device)
        sketch = M_float.matmul(probes)
        local_q, _ = torch.linalg.qr(sketch, mode="reduced")
        projector = local_q.matmul(local_q.t()).flatten()
        mean_projector = projector.detach().clone()
        self._all_reduce_scalar(mean_projector, tp_group)
        mean_projector = mean_projector / float(tp_group.size())
        dist = (projector - mean_projector).norm()
        return float((dist / math.sqrt(float(q))).item())

    def _monitor_signal_value(
        self,
        M: torch.Tensor,
        tp_group: torch.distributed.ProcessGroup,
    ) -> float:
        if self.snecv_monitor_signal == "energy_cv":
            return self._energy_cv(M, tp_group)
        if self.snecv_monitor_signal == "stable_rank_cv":
            return self._stable_rank_cv(M, tp_group)
        if self.snecv_monitor_signal == "spectral_norm_cv":
            return self._spectral_norm_cv(M, tp_group)
        if self.snecv_monitor_signal == "directional_gram_cv":
            return self._directional_gram_cv(M, tp_group)
        if self.snecv_monitor_signal == "gram_sketch_distance":
            return self._gram_sketch_distance(M, tp_group)
        if self.snecv_monitor_signal == "subspace_angle_sketch":
            return self._subspace_angle_sketch(M, tp_group)
        raise ValueError(f"Unsupported monitor signal {self.snecv_monitor_signal!r}")

    def _snecv_update_and_score(
        self,
        p: torch.Tensor,
        M: torch.Tensor,
        tp_group: torch.distributed.ProcessGroup,
    ) -> tuple[float, float, float, float]:
        """
        Returns:
            raw_cv, mean_cv, var_cv, z_score

        z_score is computed against the PREVIOUS running mean/variance,
        then the running statistics are updated using the current raw_cv.
        """
        raw_cv = self._monitor_signal_value(M, tp_group)
        meta = self._get_meta(p)

        prev_mean = float(meta.get("snecv_mean", raw_cv))
        prev_var = float(meta.get("snecv_var", 0.0))
        prev_count = int(meta.get("snecv_count", 0))

        if prev_count < 2:
            z = 0.0
        else:
            z = (raw_cv - prev_mean) / math.sqrt(prev_var + self.snecv_eps)

        beta = self.snecv_beta
        if prev_count == 0:
            new_mean = raw_cv
            new_var = 0.0
        else:
            delta = raw_cv - prev_mean
            new_mean = beta * prev_mean + (1.0 - beta) * raw_cv
            new_var = beta * prev_var + (1.0 - beta) * (delta * delta)

        meta["snecv_mean"] = float(new_mean)
        meta["snecv_var"] = float(new_var)
        meta["snecv_count"] = int(prev_count + 1)
        meta["snecv_raw_cv"] = float(raw_cv)
        meta["snecv_z"] = float(z)

        return float(raw_cv), float(new_mean), float(new_var), float(z)

    def _current_full_threshold(self) -> float:
        if self.comm_budget_rho is None:
            return self.snecv_z_high
        return self._budget_threshold

    def _update_leaky_pressure(
        self,
        p: torch.Tensor,
        z: float,
        threshold: float,
    ) -> tuple[float, bool]:
        meta = self._get_meta(p)
        prev_pressure = float(meta.get("snecv_pressure", 0.0))
        pressure = self.snecv_pressure_gamma * prev_pressure

        in_medium_band = self.snecv_z_low <= z < threshold
        if in_medium_band:
            normalized = (z - self.snecv_z_low) / (threshold - self.snecv_z_low + self.snecv_eps)
            pressure += 1.0 + self.snecv_pressure_alpha * normalized

        return float(pressure), bool(in_medium_band)

    def _update_budget_threshold(self, used_full: bool) -> None:
        """
        Online threshold adaptation to target the long-run full-step budget rho.

        This is a simple Robbins-Monro style update:
            tau <- tau + eta_t * (I[full] - rho)
        with eta_t = 1 / sqrt(t).
        """
        if self.comm_budget_rho is None:
            return

        self._budget_updates += 1
        eta_t = 1.0 / math.sqrt(float(self._budget_updates))
        indicator = 1.0 if used_full else 0.0

        self._budget_threshold += eta_t * (indicator - self.comm_budget_rho)

        # keep the threshold in a sane range
        self._budget_threshold = max(
            self.snecv_z_low + 1e-6,
            min(self._budget_threshold, 10.0),
        )

    def _get_local_lr_scale(self, z: float, threshold: float) -> float:
        """Return the blockwise LR multiplier for the current z-score regime."""
        if z < self.snecv_z_low or z >= threshold:
            return 1.0

        if self.snecv_use_smooth_local_lr_decay:
            penalty = max(0.0, z - self.snecv_z_low)
        else:
            penalty = max(0.0, z)

        return 1.0 / (1.0 + self.snecv_local_lr_gamma * penalty)

    def _scaled_orthogonalize(
        self,
        grad: torch.Tensor,
        tp_group: torch.distributed.ProcessGroup | None,
        partition_dim: int | None,
        ns_mode: Literal["duplicated", "distributed"] = "duplicated",
    ) -> torch.Tensor:
        size0, size1 = int(grad.size(-2)), int(grad.size(-1))
        size = [size0, size1]
        if partition_dim is not None and tp_group is not None:
            size[partition_dim] *= tp_group.size()

        log_single_rank(
            logger,
            logging.DEBUG,
            f"Orthogonalizing grad: steps={self.num_ns_steps}, coeff={self.coefficient_type}, "
            f"scale_mode={self.scale_mode}, extra_scale_factor={self.extra_scale_factor}, "
            f"partition_dim={partition_dim}, ns_mode={ns_mode}",
        )

        orth_grad = newton_schulz_tp(
            grad,
            steps=self.num_ns_steps,
            coefficient_type=self.coefficient_type,
            tp_group=tp_group,
            partition_dim=partition_dim,
            mode=ns_mode,
        )

        scale_factor = get_muon_scale_factor(size[0], size[1], mode=self.scale_mode)
        return orth_grad * scale_factor * self.extra_scale_factor

    def _get_tp_group(self, p: torch.Tensor) -> torch.distributed.ProcessGroup | None:
        if self.pg_collection:
            return self.pg_collection.expt_tp if getattr(p, "expert_tp", False) else self.pg_collection.tp
        return None

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        tp_group = self._get_tp_group(p)
        meta = self._get_meta(p)

        param_partition_dim = getattr(p, "partition_dim", None)
        if param_partition_dim == -1:
            param_partition_dim = None

        can_switch = (
            self.mode == "blockwise"
            and tp_group is not None
            and tp_group.size() > 1
            and param_partition_dim is not None
        )

        step = int(self._muon_global_step)
        use_full = False
        local_lr_scale = 1.0
        full_reason: str | None = None

        raw_cv = 0.0
        z = 0.0
        pressure = float(meta.get("snecv_pressure", 0.0))
        tau_high = self._current_full_threshold()
        decision_threshold = tau_high

        if can_switch:
            if step >= self.snecv_warmup_steps:
                raw_cv, _, _, z = self._snecv_update_and_score(p, grad, tp_group)
                decision_threshold = tau_high
            
                local_lr_scale = self._get_local_lr_scale(z, tau_high)
                if z >= tau_high:
                    use_full = True
                    full_reason = "z"
                    pressure = self.snecv_pressure_reset_factor * pressure
                else:
                    pressure, _ = self._update_leaky_pressure(p, z, tau_high)
                    if pressure >= self.snecv_pressure_threshold_h:
                        use_full = True
                        full_reason = "pressure"
                        pressure = self.snecv_pressure_reset_factor * pressure

                self._update_budget_threshold(use_full)
                tau_high = self._current_full_threshold()

            meta["snecv_threshold"] = float(tau_high)
            meta["snecv_pressure"] = float(pressure)
            meta["snecv_pressure_threshold"] = float(self.snecv_pressure_threshold_h)

        meta["used_full_orth"] = bool(use_full)
        meta["local_lr_scale"] = float(local_lr_scale)
        meta["snecv_raw_cv"] = float(raw_cv)
        meta["snecv_z"] = float(z)
        meta["snecv_full_reason"] = full_reason
        if can_switch:
            self._record_snecv_decision(
                z=z,
                threshold=decision_threshold,
                use_full=use_full,
                full_reason=full_reason,
            )

        log_single_rank(
            logger,
            logging.DEBUG,
            f"[SNECV-Muon] step={step} signal={self.snecv_monitor_signal} "
            f"raw={raw_cv:.4f} z={z:.4f} "
            f"z_low={self.snecv_z_low:.4f} z_high={tau_high:.4f} "
            f"pressure={pressure:.4f} H={self.snecv_pressure_threshold_h:.4f} "
            f"use_full={use_full} reason={full_reason} local_lr_scale={local_lr_scale:.4f}",
        )

        def run_one(mat: torch.Tensor) -> torch.Tensor:
            if use_full:
                partition_dim = param_partition_dim
                ns_mode = "distributed"
            else:
                partition_dim = None
                ns_mode = "duplicated"

            return self._scaled_orthogonalize(mat, tp_group, partition_dim, ns_mode)

        if self.split_qkv and self.is_qkv_fn and self.is_qkv_fn(p):  # type: ignore[misc]
            grad_shape = grad.shape
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),
                self.qkv_split_shapes,
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]
            qkv_grads = [run_one(g).view(num_query_groups, -1, grad_shape[-1]) for g in qkv_grads]
            out = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            out = run_one(grad)

        return out

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        self._muon_global_step += 1
        self._snecv_step_counts = self._new_snecv_counter_dict()

        for group in self.param_groups:
            lr_base = group["lr"]
            lr_block = group.get("lr_block", lr_base)
            lr_full = group.get("lr_full", lr_base)

            for p in group["params"]:
                if p.dim() == 1:
                    raise ValueError(f"{self.__class__.__name__} does not support 1D parameters")

                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                meta = self._get_meta(p)

                used_full = bool(meta.get("used_full_orth", False))
                local_lr_scale = float(meta.get("local_lr_scale", 1.0))
                lr_eff = lr_full if used_full else (lr_block * local_lr_scale)

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                exp_avg = state["momentum_buffer"]

                self._apply_weight_decay_inplace(
                    p,
                    g,
                    lr_eff,
                    group["weight_decay"],
                )

                exp_avg.lerp_(g, 1 - group["momentum_beta"])

                if self.use_nesterov:
                    upd = g.lerp(exp_avg, group["momentum_beta"])
                else:
                    upd = exp_avg

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    group_kwargs = {k: v for k, v in group.items() if k != "params"}
                    upd = self.orthogonalize(p, upd, **group_kwargs)

                self.pre_weight_update_fn_inplace(p, upd)
                p.add_(upd, alpha=-lr_eff)
                self.post_weight_update_fn_inplace(p)

        self._log_snecv_frequency_stats()
        return loss


def get_megatron_muon_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """This function is used to get the muon optimizer for the model chunks.
    It is used to get the muon optimizer for the model chunks.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        layer_wise_distributed_optimizer (bool): if true, use layer-wise distributed optimizer.
            Defaults to False.
    """
    # TODO: Mutating config.optimizer is a side effect; clean up after
    # https://github.com/NVIDIA/Megatron-LM/pull/3638 lands.
    # Set the nonlinear optimizer for muon (used for embeddings, biases, norms).
    config.optimizer = config.muon_scalar_optimizer

    assert HAVE_EMERGING_OPTIMIZERS, "Emerging Optimizers >= 0.2 is not installed."

    # Dist-opt is not supported due to strong coupling with how DDP init grad buffer
    # In theory we can change DDP to enable use muon and dist-opt-adam together
    if config.use_distributed_optimizer:
        raise Exception('muon with dist optimizer is not supported.')
    # only support bf16 w/o loss scale now
    if config.fp16:
        raise Exception('muon with fp16 is not supported.')

    # before this function receive properly created collection
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up emerging optimizer with config {config}')

    # Needed for torch_dist ckpt_format, unlike torch ckpt_format
    # For other emerging optimizers, need to implement init_state_fn as well
    # TODO(boxiangw): Improve usability after optimizer refactor
    # TODO(boxiangw): support precision aware optimizer
    def muon_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    def lion_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)

    nonlinear_init_state_fn = (
        lion_init_state_fn if config.muon_scalar_optimizer == 'lion' else adam_init_state_fn
    )

    optimizers = []
    # record list of non/linear params
    linear_params = []
    nonlinear_params = []
    for model_chunk in model_chunks:
        # use config to determine qkv split shapes.
        # no need to check tp since tp splits by head and this is per head(group) dimension
        num_attention_heads = model_chunk.config.num_attention_heads
        num_query_groups = model_chunk.config.num_query_groups
        kv_channels = model_chunk.config.kv_channels
        qkv_split_shapes = [
            num_attention_heads // num_query_groups * kv_channels,
            kv_channels,
            kv_channels,
        ]
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            # add flag for expert weight so optimizer can figure which tp group it uses
            # alternatively, create new param group and save tp_group. this require more
            # change in optimizer
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # add flag for qkv parameter
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            # TODO(deyuf): currently only allow 2D non-embedding weight to avoid breaking
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    muon_kwargs = {
        "lr": config.lr,
        "momentum_beta": config.muon_momentum,
        "use_nesterov": config.muon_use_nesterov,
        "weight_decay": config.weight_decay,
        "fp32_matmul_prec": config.muon_fp32_matmul_prec,
        "coefficient_type": getattr(config, "muon_coefficient_type", "simple"),
        "num_ns_steps": config.muon_num_ns_steps,
        "scale_mode": config.muon_scale_mode,
        "split_qkv": config.muon_split_qkv,
        "is_qkv_fn": lambda p: getattr(p, "is_qkv", False),
        "qkv_split_shapes": qkv_split_shapes,
        "extra_scale_factor": config.muon_extra_scale_factor,
        "pg_collection": pg_collection,
        # 构造函数只接受 "blockwise"，选配逻辑用单独变量
        "mode": "blockwise",
        # 以下是安全的 int/float 默认值，每个 config 分支会覆盖需要的项
        "lr_block": config.lr,
        "lr_full": config.lr,
        "snecv_beta": getattr(config, "muon_snecv_beta", 0.98),
        "snecv_eps": getattr(config, "muon_snecv_eps", 1e-6),
        "snecv_z_low": 1.0,
        "snecv_z_high": 3.0,
        # 默认值从 0.95 (float) 改为 200 (int)
        "snecv_warmup_steps": 200,
        "snecv_pressure_gamma": 0.95,
        "snecv_pressure_alpha": 1.0,
        "snecv_pressure_threshold_h": 4.0,
        "snecv_pressure_reset_factor": 0.0,
        "snecv_local_lr_gamma": 0.5,
        "snecv_use_smooth_local_lr_decay": True,
        "snecv_monitor_signal": "energy_cv",
        "snecv_monitor_sketch_q": 4,
        "snecv_monitor_power_iters": 2,
        "snecv_stats_log_interval": getattr(config, "muon_snecv_stats_log_interval", 1),
        "comm_budget_rho": None,
    }

    # 把 muon_config_mode 仅当作 config 选择器，不传给构造函数
    muon_config_mode = getattr(config, "muon_config_mode", "blockwise")

    # Config 1: Blockwise Muon
    # 目标: TP 切分时完全不进行通信，每个 rank 仅对本地 shard 做 NS iteration
    #
    # 实现原理:
    #   snecv_warmup_steps 设为极大值 → 整个训练过程中
    #     if step >= snecv_warmup_steps 永远为 False
    #   → 跳过整个 SNECV 评分块
    #   → use_full 保持默认 False
    #   → run_one() 中 partition_dim=None, ns_mode="duplicated"
    #   → newton_schulz_tp 在各 rank 上独立处理本地 shard，零通信
    #
    # 附加: 把 monitor 参数调到最小值，万一漏进 SNECV 块也不做多余计算
    if muon_config_mode == "blockwise":
        muon_kwargs.update({
            "lr_block": config.lr,
            "lr_full": config.lr,
            # 永远处于 warmup → SNECV 逻辑永不触发 → 永远 blockwise
            "snecv_warmup_steps": 999_999_999,
            # 以下仅作为安全后备（warmup 内根本不会被读到）
            "snecv_z_low": 1.0,
            "snecv_z_high": 9999.0,
            "snecv_pressure_gamma": 0.0,
            "snecv_pressure_threshold_h": 9999.0,
            "snecv_local_lr_gamma": 0.0,        # 不衰减 blockwise LR
            "snecv_monitor_signal": "energy_cv", # 最轻量 monitor
            "snecv_monitor_sketch_q": 1,
            "snecv_monitor_power_iters": 0,
            "comm_budget_rho": None,
        })

    # Config 2: SNECV-Muon  (monitor sweep 实验)
    # 目标: 自适应切换 blockwise / full，跑不同 monitor signal 的 sweep
    #
    # 实现原理:
    #   warmup 结束后，每步计算 monitor signal → EWMA z-score →
    #     z >= tau_high        → full orth (distributed NS + allgather)
    #     z_low <= z < tau_high → 累积 leaky pressure；若 pressure >= H → full
    #     z < z_low            → blockwise (local NS，零通信)
    #
    # 外部 config 可控的 sweep 维度:
    #   muon_snecv_monitor_signal   ∈ {energy_cv, stable_rank_cv, spectral_norm_cv,
    #                                   directional_gram_cv, gram_sketch_distance,
    #                                   subspace_angle_sketch}
    #   muon_snecv_monitor_sketch_q, muon_snecv_monitor_power_iters
    #   muon_lr_block / muon_lr_full, 以及所有 SNECV 超参
    elif muon_config_mode == "snecv":
        muon_kwargs.update({
            "lr_block": getattr(config, "muon_lr_block", config.lr),
            "lr_full": getattr(config, "muon_lr_full", config.lr),
            "snecv_beta": getattr(config, "muon_snecv_beta", 0.98),
            "snecv_z_low": getattr(config, "muon_snecv_z_low", 1.0),
            "snecv_z_high": getattr(config, "muon_snecv_z_high", 3.0),
            "snecv_warmup_steps": getattr(config, "muon_snecv_warmup_steps", 200),
            "snecv_pressure_gamma": getattr(config, "muon_snecv_pressure_gamma", 0.95),
            "snecv_pressure_alpha": getattr(config, "muon_snecv_pressure_alpha", 1.0),
            "snecv_pressure_threshold_h": getattr(
                config, "muon_snecv_pressure_threshold_h", 4.0
            ),
            "snecv_pressure_reset_factor": getattr(
                config, "muon_snecv_pressure_reset_factor", 0.0
            ),
            "snecv_local_lr_gamma": getattr(config, "muon_snecv_local_lr_gamma", 0.5),
            "snecv_use_smooth_local_lr_decay": True,
            "snecv_monitor_signal": getattr(
                config, "muon_snecv_monitor_signal", "energy_cv"
            ),
            "snecv_monitor_sketch_q": getattr(
                config, "muon_snecv_monitor_sketch_q", 4
            ),
            "snecv_monitor_power_iters": getattr(
                config, "muon_snecv_monitor_power_iters", 2
            ),
            "comm_budget_rho": getattr(config, "muon_comm_budget_rho", None),
        })

    # Config 3: 伪装成 Original (Full Update) Muon 的基线
    # 目标: 每步都做 full distributed Newton-Schulz iteration + 全通信
    #       等价于原始 Muon 的行为
    #
    # 实现原理 (利用 pressure 机制让 use_full 每步为 True):
    #   warmup_steps=0        → 第 1 步就进入 SNECV 评分块
    #   z_low=0.0, z_high=999 → z-score 几乎不可能 >=999，走 pressure 路径
    #   pressure_gamma=1.0    → pressure 永不衰减
    #   pressure_alpha=0.0    → medium band 内每步固定加 1.0
    #   pressure_threshold_h=1e-5 → 第 1 步 pressure=1.0 即远超阈值 → full
    #   pressure_reset_factor=1.0 → full 触发后 pressure 不重置
    #   → 此后 pressure 只增不减，每步都满足 >= 1e-5 → 永远 use_full=True
    #
    # 注意: 每步有一次轻量的 energy_cv all_reduce (2 个 scalar)，
    #       相比 distributed NS 的 allgather/reduce_scatter 可忽略不计。
    else:
        muon_kwargs.update({
            "lr_block": config.lr,
            "lr_full": config.lr,
            "snecv_warmup_steps": 0,
            "snecv_z_low": 0.0,
            "snecv_z_high": 999.0,
            "snecv_pressure_gamma": 1.0,           # 不衰减
            "snecv_pressure_alpha": 0.0,            # 每步固定 +1.0
            "snecv_pressure_threshold_h": 1e-5,     # 极小，第 1 步即触发
            "snecv_pressure_reset_factor": 1.0,     # 触发后不重置
            "snecv_local_lr_gamma": 0.0,            # 无 LR 衰减（反正 use_full=True 时不用）
            "snecv_monitor_signal": "energy_cv",    # 最轻量 monitor
            "snecv_monitor_sketch_q": 1,
            "snecv_monitor_power_iters": 0,
            "comm_budget_rho": None,                # 禁用 budget 自适应
        })

    # freezing nonlinear params and get param groups for muon
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)
    # if layerwise distributed optimizer is not used, need to handle ep params separately
    expert_param_groups = []
    if not layer_wise_distributed_optimizer:
        for group in linear_param_groups:
            if group['is_expert_parallel']:
                expert_param_groups.append(group)
                linear_param_groups.remove(group)

    optimizer = AdaptiveTensorParallelMuon(linear_param_groups, **muon_kwargs)

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            # creating master weight before layerwise sharding will lead to unnecessary master
            # weight so here we delay master weight creation into layer_wise unset config.bf16
            # will also result in all optimizers below(adam) to also not be wrapped
            config.bf16 = False
            reset_config_bf16 = True
        else:
            # if not using layer_wise wrapper, just create master weight here is fine
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer, config, None, muon_init_state_fn
            )
    else:
        optimizer = FP32Optimizer(optimizer, config, muon_init_state_fn)

    optimizers.append(optimizer)

    # expert optimizer exists meaning layerwise distributed optimizer is not used
    if len(expert_param_groups) > 0:
        expert_optimizer = AdaptiveTensorParallelMuon(expert_param_groups, **muon_kwargs)
        if config.bf16:
            expert_optimizer = Float16OptimizerWithFloat16Params(
                expert_optimizer, config, None, muon_init_state_fn
            )
        else:
            expert_optimizer = FP32Optimizer(expert_optimizer, config, muon_init_state_fn)
        setattr(expert_optimizer, 'grad_stats_parallel_group', pg_collection.tp_ep_pp)
        optimizers.append(expert_optimizer)

    # done with muon, unfreeze nonlinear and freeze linear
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # call original get. linear params will be skipped since they're freezed
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )

    # unfreeze everything
    for param in linear_params:
        param.requires_grad = True

    # chain everything together
    init_fns = [muon_init_state_fn] + len(chained_adam.chained_optimizers) * [
        nonlinear_init_state_fn
    ]
    optimizers += chained_adam.chained_optimizers

    if layer_wise_distributed_optimizer:
        log_single_rank(logger, logging.INFO, 'Using LayerWiseDistributedOptimizer for Muon')
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(
            optimizers,
            config,
            pg_collection,
            init_state_fn_list=init_fns,
            model_chunks=model_chunks,
            async_allgather=config.overlap_param_gather,
        )
    return ChainedOptimizer(optimizers)
