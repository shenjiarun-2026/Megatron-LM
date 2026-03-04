# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron muon optimizer wrapper to handle tensor-parallel."""

import logging
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_size, log_single_rank

from . import _get_param_groups, get_megatron_optimizer
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey

try:
    from emerging_optimizers import utils
    from emerging_optimizers.orthogonalized_optimizers import (
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object


logger = logging.getLogger(__name__)


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
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        if torch.distributed.get_rank() == 0:
            print(
                f'Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )

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
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                mode="duplicated" if mode == "blockwise" else mode,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.mode = mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        super().__init__(
            params,
            lr,
            momentum_beta,
            use_nesterov=use_nesterov,
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


class AdaptiveTensorParallelMuon (OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer with adaptive local/full orthogonalization triggers."""

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
        adaptive_full: bool = True,
        diag_mode: Literal["none", "energy_cv", "coherence", "hybrid"] = "hybrid",
        diag_k: int = 2,
        cv_threshold: float = 0.6,
        coherence_threshold: float = 0.75,
        ns_residual_threshold: float = 0.15,
        full_cooldown: int = 20,
        diag_seed: int = 12345,
        lr_block: float | None = None,
        lr_full: float | None = None,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if diag_k < 1:
            raise ValueError(f"diag_k must be >= 1, got {diag_k}")

        self.pg_collection = pg_collection
        self.mode = mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

        # store muon configs so we can run local/full dynamically
        self.fp32_matmul_prec = fp32_matmul_prec
        self.coefficient_type = coefficient_type
        self.num_ns_steps = num_ns_steps
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor

        # adaptive trigger configs
        self.adaptive_full = adaptive_full
        self.diag_mode = diag_mode
        self.diag_k = diag_k
        self.cv_threshold = cv_threshold
        self.coherence_threshold = coherence_threshold
        self.ns_residual_threshold = ns_residual_threshold
        self.full_cooldown = full_cooldown
        self.diag_seed = diag_seed

        # global step counter (incremented once per optimizer.step)
        self._muon_global_step = 0

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"

        # NOTE: pass lr_block/lr_full through param_group as extra kwargs (OrthogonalizedOptimizer allows **kwargs)
        super().__init__(
            params,
            lr,
            momentum_beta,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=None,  # we override orthogonalize anyway
            lr_block=lr_block,
            lr_full=lr_full,
        )

    # low-comm diagnostics
    @staticmethod
    def _all_reduce_scalar(x: torch.Tensor, group: torch.distributed.ProcessGroup) -> torch.Tensor:
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM, group=group)
        return x

    def _energy_cv(self, M: torch.Tensor, tp_group: torch.distributed.ProcessGroup) -> float:
        """Shard energy coefficient-of-variation across TP ranks (scalar comm)."""
        eps = 1e-12
        e = (M.float() * M.float()).sum()  # scalar
        e2 = e * e
        e_sum = e.detach().clone()
        e2_sum = e2.detach().clone()
        self._all_reduce_scalar(e_sum, tp_group)
        self._all_reduce_scalar(e2_sum, tp_group)

        world = float(tp_group.size())
        mean = e_sum / world
        var = (e2_sum / world) - mean * mean
        var = torch.clamp(var, min=0.0)
        cv = (torch.sqrt(var) / (mean + eps)).item()
        return float(cv)

    def _coherence_score(self, M: torch.Tensor, tp_group: torch.distributed.ProcessGroup, k: int) -> float:
        """
        Cross-rank direction consistency via scalar sketch:
            s_i = |sum_r a_{r,i}| / sum_r |a_{r,i}|
        where a_{r,i} = w_i^T (M_r z_i).
        Communication: O(k) scalars (2 all-reduce per probe).
        """
        eps = 1e-12
        m, n = M.shape[-2], M.shape[-1]
        device = M.device
        dtype = torch.float32

        # Use deterministic seeds so all ranks sample the same probes.
        # Important: decision must match on every rank.
        score_acc = 0.0
        base = self.diag_seed + 1315423911 * int(self._muon_global_step)

        M32 = M.float()

        for i in range(k):
            g = torch.Generator(device=device)
            g.manual_seed(base + i * 97)

            z = torch.randn((n,), generator=g, device=device, dtype=dtype)
            w = torch.randn((m,), generator=g, device=device, dtype=dtype)

            # a = w^T (M z)
            a_local = torch.dot(w, torch.mv(M32, z))  # scalar float32

            sum_a = a_local.detach().clone()
            sum_abs = a_local.detach().abs().clone()

            self._all_reduce_scalar(sum_a, tp_group)
            self._all_reduce_scalar(sum_abs, tp_group)

            s_i = (sum_a.abs() / (sum_abs + eps)).item()
            score_acc += float(s_i)

        return score_acc / float(k)

    def _ns_residual_proxy(self, Q: torch.Tensor, k: int = 1) -> float:
        """
        Cheap proxy of orthogonality error:
            r ≈ || Q^T(Q z) - z || / ||z||
        If Q has orthonormal columns, Q^T Q = I, residual ~ 0.
        No communication here; for triggering we will take TP max later.
        """
        eps = 1e-12
        m, n = Q.shape[-2], Q.shape[-1]
        device = Q.device
        Q32 = Q.float()

        base = self.diag_seed + 2654435761 * int(self._muon_global_step)
        g = torch.Generator(device=device)
        g.manual_seed(base)

        z = torch.randn((n, k), generator=g, device=device, dtype=torch.float32)
        y = Q32 @ z                  # (m, k)
        z2 = Q32.mT @ y              # (n, k)
        diff = z2 - z
        r = (diff.norm() / (z.norm() + eps)).item()
        return float(r)

    # orthogonalization core
    def _scaled_orthogonalize(
        self,
        grad: torch.Tensor,
        tp_group: torch.distributed.ProcessGroup | None,
        partition_dim: int | None,
        *,
        ns_mode: Literal["duplicated", "distributed"] = "duplicated",
    ) -> torch.Tensor:
        """
        Run Newton–Schulz orthogonalization (local or full depending on partition_dim/ns_mode),
        then apply Muon scaling.
        """
        size0, size1 = int(grad.size(-2)), int(grad.size(-1))
        size = [size0, size1]
        if partition_dim is not None and tp_group is not None:
            size[partition_dim] *= tp_group.size()

        log_single_rank(
            logger,
            logging.INFO,
            f"Orthogonalizing grad: steps={self.num_ns_steps}, coeff={self.coefficient_type}, "
            f"scale_mode={self.scale_mode}, extra_scale_factor={self.extra_scale_factor}, "
            f"partition_dim={partition_dim}, ns_mode={ns_mode}",
        )

        # newton_schulz_tp expects tp_group if partition_dim is not None
        if partition_dim is None or tp_group is None:
            # local path (no comm)
            orth_grad = newton_schulz_tp(
                grad,
                steps=self.num_ns_steps,
                coefficient_type=self.coefficient_type,
                tp_group=None,  # ignored when partition_dim is None inside newton_schulz_tp fallback
                partition_dim=None,
                mode="duplicated",
            )
        else:
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
        """
        Orthogonalize the (momentum) update.
        In blockwise mode, optionally switch between local (no comm) and full (gather-cat) orthogonalization
        based on low-comm diagnostics.
        """
        tp_group = self._get_tp_group(p)
        state = self.state[p]

        # Read TP partition_dim if present
        param_partition_dim = getattr(p, "partition_dim", None)
        if param_partition_dim == -1:
            param_partition_dim = None

        # Default behavior: respect optimizer mode
        do_adaptive = (
            self.adaptive_full
            and self.mode == "blockwise"
            and tp_group is not None
            and tp_group.size() > 1
            and param_partition_dim is not None
        )

        # In pure blockwise: local orth => partition_dim=None (no comm).
        # In full step: use partition_dim and ns_mode="duplicated" to all_gather the full matrix within tp_group.
        use_full = False

        if do_adaptive and self.diag_mode != "none":
            cooldown_until = int(state.get("cooldown_until_step", 0))
            if self._muon_global_step >= cooldown_until:
                # 1) energy CV (scalar comm)
                cv = 0.0
                if self.diag_mode in ("energy_cv", "hybrid"):
                    cv = self._energy_cv(grad, tp_group)

                # 2) coherence score (scalar comm, O(k))
                coh = 1.0
                if self.diag_mode in ("coherence", "hybrid"):
                    coh = self._coherence_score(grad, tp_group, k=self.diag_k)

                # 3) previous NS residual proxy (use TP max to make it globally consistent)
                prev_r = float(state.get("ns_residual", 0.0))
                prev_r_t = torch.tensor(prev_r, device=grad.device, dtype=torch.float32)
                torch.distributed.all_reduce(prev_r_t, op=torch.distributed.ReduceOp.MAX, group=tp_group)
                prev_r_max = float(prev_r_t.item())

                # trigger rule (simple, debuggable)
                if (cv > self.cv_threshold) or (coh < self.coherence_threshold) or (prev_r_max > self.ns_residual_threshold):
                    use_full = True
                    state["cooldown_until_step"] = int(self._muon_global_step + self.full_cooldown)

                # optional debug logging
                log_single_rank(
                    logger,
                    logging.DEBUG,
                    f"[MuonDiag] step={self._muon_global_step} cv={cv:.4f} coh={coh:.4f} prev_r_max={prev_r_max:.4f} "
                    f"-> use_full={use_full}",
                )

        # record for two-lr step
        state["used_full_orth"] = bool(use_full)

        # ---------- orthogonalization (support split_qkv) ----------
        def run_one(mat: torch.Tensor) -> torch.Tensor:
            if use_full:
                return self._scaled_orthogonalize(mat, tp_group, param_partition_dim, ns_mode="duplicated")
            # local blockwise
            return self._scaled_orthogonalize(mat, tp_group, partition_dim=None, ns_mode="duplicated")

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

        # update NS residual proxy for next-step triggering
        # (compute locally; triggering uses TP max of previous value so ranks stay consistent)
        try:
            state["ns_residual"] = self._ns_residual_proxy(out, k=1)
        except Exception:
            # never kill training for diagnostics
            state["ns_residual"] = float(state.get("ns_residual", 0.0))

        return out

    # two-step-size step()
    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Same as OrthogonalizedOptimizer.step, but uses lr_block/lr_full depending on whether
        this parameter used full orthogonalization in this step (MuonBP-style two stepsizes).
        """
        loss = None if closure is None else closure()
        self._muon_global_step += 1

        for group in self.param_groups:
            lr_base = group["lr"]
            lr_block = group.get("lr_block", None)
            lr_full = group.get("lr_full", None)
            lr_block = lr_base if lr_block is None else lr_block
            lr_full = lr_base if lr_full is None else lr_full

            for p in group["params"]:
                if p.dim() == 1:
                    raise ValueError(f"{self.__class__.__name__} does not support 1D parameters")
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                exp_avg = state["momentum_buffer"]

                # weight decay (same as base)
                self._apply_weight_decay_inplace(
                    p,
                    g,
                    lr_base,
                    group["weight_decay"],
                )

                # momentum EMA
                exp_avg.lerp_(g, 1 - group["momentum_beta"])

                # nesterov or plain momentum
                if self.use_nesterov:
                    upd = g.lerp(exp_avg, group["momentum_beta"])
                else:
                    upd = exp_avg

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    group_kwargs = {k: v for k, v in group.items() if k != "params"}
                    upd = self.orthogonalize(p, upd, **group_kwargs)

                # choose lr based on whether we used full orth
                used_full = bool(state.get("used_full_orth", False))
                lr_eff = lr_full if used_full else lr_block

                p.add_(upd, alpha=-lr_eff)

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
    # Muon currently use adam config. setting str here to call regular get for adam creation
    # side effect is muon optimizer will have wrong name, i.e. config.optimizer == 'adam'
    config.optimizer = 'adam'

    assert HAVE_EMERGING_OPTIMIZERS, "Emerging Optimizers is not installed."

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
        "num_ns_steps": config.muon_num_ns_steps,
        "scale_mode": config.muon_scale_mode,
        "split_qkv": config.muon_split_qkv,
        "is_qkv_fn": lambda p: getattr(p, "is_qkv", False),
        "qkv_split_shapes": qkv_split_shapes,
        "extra_scale_factor": config.muon_extra_scale_factor,
        "pg_collection": pg_collection,
        "mode": config.muon_tp_mode,
        "coefficient_type": "simple",
    }

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

    optimizer = TensorParallelMuon(linear_param_groups, **muon_kwargs)

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
        expert_optimizer = TensorParallelMuon(expert_param_groups, **muon_kwargs)
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
    init_fns = [muon_init_state_fn] + len(chained_adam.chained_optimizers) * [adam_init_state_fn]
    optimizers += chained_adam.chained_optimizers

    if layer_wise_distributed_optimizer:
        log_single_rank(logger, logging.INFO, 'Using LayerWiseDistributedOptimizer for Muon')
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(
            optimizers, config, pg_collection, init_state_fn_list=init_fns
        )
    return ChainedOptimizer(optimizers)
