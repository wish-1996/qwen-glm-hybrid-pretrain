# 10 对齐 DeepSeek-V4：差距清单与实现路线图

本文档用于把 DeepSeek-V4 技术报告的关键工程点，映射到本仓库当前实现，形成一份**可逐步打勾**的对齐路线图。

> 约定：
> - **本仓库当前状态**：以 `fix/dataloader-prod` 分支为准。
> - **优先级**：P0（必须先做）/ P1（长上下文训练可行性）/ P2（系统级工程，重投入）。

---

## 0. 我们当前已经具备的基础（对齐点）

- 训练主入口清晰：`train/train_multimodal.py`（DDP + checkpoint + JSONL metrics）
- MoE 已具备"生产化后端"切换能力：
  - `MOE_BACKEND=native|deepspeed`
  - `model/moe_deepspeed.py`（DeepSpeed-MoE wrapper）
- 注意力侧提供可选加速入口：
  - `use_flash_attn / attention_backend`
  - flash-attn 不可用时自动回退到 torch 路径
- 线性注意力（GatedDeltaNet）已识别为长序列瓶颈，并已规划/实现优化路径（chunk-wise → scan kernel）
- smoke 回归能力（关键开关可跑通）：
  - `scripts/smoke_1node_1gpu.sh`
  - （如已加入）`scripts/smoke_1node_8gpu.sh`

---

## 1. DeepSeek-V4 核心优势拆解（按层次）

> 这里按"最影响 long-context 的关键路径"排序：注意力/KV → MoE/并行 → 训练稳定性 → 精度与系统工程。

### Level-1：长上下文核心（CSA/HCA + KV 体系）

DeepSeek-V4 关键点（来自技术报告）：
- **Hybrid Attention**：Compressed Sparse Attention（CSA） + Heavily Compressed Attention（HCA）
- **sliding window** 分支用于补近邻信息（V4 里是 128）
- **Partial RoPE**：只对部分维度做 RoPE（V4 配置里 `qk_rope_head_dim=64`）
- **Q/K 与 compressed KV 的额外归一化**，以及 attention sink 等稳定性技巧
- **KV cache 极致优化**：混合精度存储、异构 KV 布局、甚至 on-disk KV cache

本仓库现状：
- ✅ 有 StandardAttention + GatedDeltaNet 两条 attention 路径
- ✅ 有 RoPE scaling（linear/ntk/dynamic_ntk）与 3D M-RoPE
- ❌ 没有 CSA/HCA（KV 压缩 + 稀疏 top-k + hybrid layer layout）
- ❌ 没有"异构 KV cache 管理/布局/落盘"相关实现

建议路线（P1→P2）：
1. **P1：先把训练侧做成 varlen/packing 友好**（为后续任何长上下文注意力方案铺路）
2. **P2（原型）：实现简化版"KV 压缩 + top-k 稀疏"注意力层**（不必一次对齐全部 CSA/HCA 细节）
3. **P2（工程）：KV cache 布局/管理 + 可能的 on-disk KV**（偏推理服务侧）

---

### Level-2：MoE 规模化（EP / all-to-all / overlap）

DeepSeek-V4 关键点：
- 大规模专家数（例如 `n_routed_experts=384`，每 token 激活多个专家）
- 强工程化 EP（Expert Parallelism）：dispatch/combine all-to-all
- **通信-计算重叠**（wave-based），尽可能把通信延迟隐藏在 GEMM/激活中

本仓库现状：
- ✅ MoE 后端可切换（native/deepspeed），已摆脱最致命 Python 双重循环瓶颈（路线B）
- ❌ 尚未进入 EP 通信-计算重叠层（仍偏"单机/单节点"思路）

建议路线（P0→P2）：
1. **P0：先保证 MoE kernel 可用且稳定**（你们已完成/接近完成）
2. **P2：再扩到 EP（多机 all-to-all）**，并以"可跑通 + 可 profiling"为第一目标
3. **P2：最后才是 overlap/fused kernel**（成本最高）

---

### Level-3：训练稳定性与收敛（Muon + mHC）

DeepSeek-V4 关键点：
- **Muon optimizer**（对大部分参数），少数模块保留 AdamW
- **mHC（Manifold-Constrained Hyper-Connections）**：升级残差连接，提高深层稳定性
- 更细粒度 activation checkpoint/autograd 扩展

本仓库现状：
- ✅ AdamW + AMP（bf16/fp16）+ grad clip + checkpoint/resume
- ❌ 未实现 Muon
- ❌ 未实现 mHC
- ⚠️ activation checkpointing 仍是 checklist 阶段

建议路线（P2）：
- 在 P0/P1 稳定后，再评估引入 Muon/mHC（否则验证成本过高且难定位问题）

---

### Level-4：精度与系统级工程（FP8/FP4/QAT + fused kernels）

DeepSeek-V4 关键点：
- KV/计算使用 FP8/FP4 等低精度路径（部分还带 QAT）
- 大量 fused kernel、确定性 kernel 库、面向服务部署的工程体系

本仓库现状：
- ❌ 暂无 FP8/FP4/QAT 路径
- ❌ 暂无 fused kernel 体系（主要在 PyTorch 层）

建议路线（P2）：
- 先从 FP8（TransformerEngine/torchao）起步；FP4/QAT 属于更后期极致优化。

---

## 2. 对齐路线图（建议的可执行 Checklist）

### P0（规模化必须先做）
- [x] MoE：接成熟 MoE 库（DeepSpeed-MoE）替换 Python 循环（`MOE_BACKEND=deepspeed`）
- [x] Attention：flash-attn 基础版接入（可选开关 + fallback）
- [ ] DeltaNet：训练分支 chunk-wise 去 token for-loop（若已合入则打勾）
- [x] Smoke：支持 `--max_steps` 快速回归

### P1（长上下文训练可行性）
- [ ] 真正的 packing + varlen（data/sequence_alignment/train 三处联动）
- [ ] attention mask 统一成 varlen 友好接口（为 flash-attn varlen & packing 做铺垫）
- [ ] 长上下文稳定性小组件：Q/K 归一化、sliding window 分支、attention sink、Partial RoPE（可逐步加）

### P2（对标 DeepSeek-V4 的系统级工程）
- [ ] 简化版 KV 压缩 + top-k 稀疏注意力原型（验证"长上下文效率"）
- [ ] EP（expert parallel）多机 all-to-all 跑通 + profiling
- [ ] ZeRO/FSDP（优化器状态/激活内存）落地
- [ ] Muon / mHC（训练稳定性与收敛加速）
- [ ] FP8/FP4/QAT（按目标场景逐步引入）

---

## 3. 建议你们后续对齐的"里程碑目标"

建议按里程碑拆：

1. **里程碑 A：32k 可训练（吞吐可接受）**
   - packing/varlen + flash-attn varlen
2. **里程碑 B：128k 可训练（稳定性不崩）**
   - 引入更多长上下文稳定性技巧（sliding window / attention sink / partial RoPE）
3. **里程碑 C：向 1M 迈进（原型）**
   - 做 KV 压缩 + 稀疏 top-k 注意力原型（向 CSA/HCA 靠近）
