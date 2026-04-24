"""
DeepSpeed-MoE 后端（P0-1）

目标：
- 用 DeepSpeed-MoE 替换本仓库的 Python for-loop MoE，实现 dispatch/combine + grouped GEMM
- 尽量保持与现有 SharedExpertMoE 相同的对外接口：forward(x) -> y，并提供 self.aux_loss

注意：
- 这里不强制要求用 deepspeed engine 启动；只要 torch.distributed 初始化（DDP）正常即可。
- 如果环境未安装 deepspeed，会给出清晰报错。
"""

from __future__ import annotations

import inspect
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist


class SwiGLU(nn.Module):
    """与 model/moe.py 中一致的专家 FFN：SwiGLU(gate, up, down)。"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        out = torch.nn.functional.silu(gate) * up
        return self.down_proj(out)


def _build_deepspeed_moe_layer(*, config) -> nn.Module:
    """
    兼容不同 deepspeed 版本的 MoE 构造参数。
    """
    try:
        from deepspeed.moe.layer import MoE as DeepSpeedMoE  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "未检测到 deepspeed（MoE 后端需要）。请先安装：\n"
            "  pip install deepspeed\n"
            "或按你们内部环境/镜像方式安装。"
        ) from e

    expert = SwiGLU(config.hidden_size, config.intermediate_size)

    # DeepSpeed-MoE 在 EP 场景下依赖 torch.distributed；这里先做最小安全校验：
    ep_size = int(getattr(config, "deepspeed_moe_ep_size", 1))
    if ep_size > 1 and (not dist.is_available() or not dist.is_initialized()):
        raise RuntimeError(
            f"deepspeed_moe_ep_size={ep_size} 需要先初始化 torch.distributed（请用 torchrun 启动并开启 --distributed）"
        )

    sig = inspect.signature(DeepSpeedMoE.__init__)
    params = sig.parameters

    # DeepSpeed-MoE 的常见参数：k, ep_size, capacity_factor, min_capacity, noisy_gate_policy, drop_tokens, use_rts ...
    kwargs: dict[str, Any] = {}
    if "k" in params:
        kwargs["k"] = int(config.top_k)
    elif "top_k" in params:
        kwargs["top_k"] = int(config.top_k)

    if "num_experts" in params:
        kwargs["num_experts"] = int(config.num_experts)

    if "ep_size" in params:
        kwargs["ep_size"] = ep_size
    if "capacity_factor" in params:
        kwargs["capacity_factor"] = float(getattr(config, "deepspeed_moe_capacity_factor", 1.0))
    if "min_capacity" in params:
        kwargs["min_capacity"] = int(getattr(config, "deepspeed_moe_min_capacity", 4))

    # 默认 gating 策略尽量温和：训练可用 RSample（若参数存在）
    if "noisy_gate_policy" in params:
        kwargs["noisy_gate_policy"] = "RSample"

    # 让溢出 token 尽量不 drop（若参数存在）
    if "drop_tokens" in params:
        kwargs["drop_tokens"] = False

    # residual routing（若参数存在）通常能提升稳定性
    if "use_rts" in params:
        kwargs["use_rts"] = True

    # DeepSpeed 的 MoE 构造通常是：MoE(hidden_size, expert, **kwargs)
    return DeepSpeedMoE(config.hidden_size, expert, **kwargs)


def _parse_moe_forward_output(out: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    兼容不同 deepspeed 版本 MoE.forward 的返回结构。
    期望拿到：(y, aux_loss)
    """
    # 常见：tuple(output, l_aux, exp_counts)
    if isinstance(out, tuple) or isinstance(out, list):
        y = out[0]
        l_aux = out[1] if len(out) > 1 and torch.is_tensor(out[1]) else torch.tensor(0.0, device=y.device)
        return y, l_aux

    # 有些版本可能返回对象
    if hasattr(out, "output"):
        y = out.output
        l_aux = getattr(out, "l_aux", None)
        if not torch.is_tensor(l_aux):
            l_aux = torch.tensor(0.0, device=y.device)
        return y, l_aux

    # 兜底：认为只返回 y
    if torch.is_tensor(out):
        return out, torch.tensor(0.0, device=out.device)

    raise TypeError(f"Unrecognized DeepSpeed MoE output type: {type(out)}")


class SharedExpertMoEDeepSpeed(nn.Module):
    """
    DeepSpeed-MoE + Shared Expert 的组合版本：
    - routed experts：DeepSpeed MoE
    - shared expert：所有 token 都走一条 SwiGLU（与原实现保持一致）
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ds_moe = _build_deepspeed_moe_layer(config=config)
        self.shared_expert = SwiGLU(config.hidden_size, config.intermediate_size)

        # 给训练脚本读取的属性（保持兼容）
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        ds_out = self.ds_moe(x)
        y_routed, l_aux = _parse_moe_forward_output(ds_out)

        y_shared = self.shared_expert(x)
        y = y_routed + y_shared

        # 与原训练脚本的读取方式兼容：layer.moe.aux_loss
        self.aux_loss = l_aux
        return y


__all__ = ["SharedExpertMoEDeepSpeed"]
