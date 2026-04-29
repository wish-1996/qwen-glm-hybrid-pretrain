"""
多模态模型训练脚本
包含训练循环、优化策略、分布式训练配置
"""

import sys
import os

import json
import math
import random
from dataclasses import asdict
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
import argparse
import time
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model.hybrid_moe_model import HybridMMMoEModel
from configs.model_config import ModelConfig
from data.multimodal_data_loader import get_data_loader
from data.multimodal_sequence_alignment import build_aligned_masks_and_labels
from model.mtp import mtp_loss_from_hidden


def _is_rank0() -> bool:
    return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


def _get_rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _setup_distributed(args) -> tuple[torch.device, int, int, int]:
    """
    生产级分布式初始化（torchrun 兼容）：
    - 设备绑定用 LOCAL_RANK
    - 通信 rank 用 RANK / WORLD_SIZE
    """
    if not args.distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 0, 1

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_process_group(backend=backend)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return device, rank, local_rank, world_size


def _maybe_import_deepspeed(enable: bool):
    if not enable:
        return None
    try:
        import deepspeed  # type: ignore
        return deepspeed
    except Exception as e:
        raise RuntimeError(
            "DeepSpeed is not installed but --deepspeed was enabled.\n"
            "Please install it first (recommended on Linux server): pip install deepspeed"
        ) from e


def _get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(
    *,
    path: str,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    config: ModelConfig,
) -> None:
    ckpt = {
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "global_step": int(global_step),
        "model_state_dict": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if getattr(scaler, "is_enabled", lambda: False)() else None,
        "rng_state": _get_rng_state(),
        "args": vars(args),
        "model_config": asdict(config),
        "saved_at": datetime.utcnow().isoformat(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def _load_checkpoint(
    *,
    path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    map_location: torch.device,
) -> dict:
    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["model_state_dict"]
    (model.module if hasattr(model, "module") else model).load_state_dict(state_dict, strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ckpt.get("scaler_state_dict") is not None and getattr(scaler, "is_enabled", lambda: False)():
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    _set_rng_state(ckpt.get("rng_state", {}))
    return ckpt


def train(args):
    """
    训练函数
    """
    deepspeed = _maybe_import_deepspeed(args.deepspeed)

    # DeepSpeed 模式：由 deepspeed.init_distributed 管理进程组；否则用我们自己的 torchrun/DDP 初始化
    if args.deepspeed:
        assert deepspeed is not None
        deepspeed.init_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        device, rank, local_rank, world_size = _setup_distributed(args)
    
    # 打印设备信息
    if _is_rank0():
        print(f"Using device: {device}, distributed={args.distributed}, rank={rank}, local_rank={local_rank}, world_size={world_size}")
    
    # 加载分词器
    if _is_rank0():
        print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if _is_rank0():
        print("Tokenizer loaded successfully.")
    
    # 加载数据
    if _is_rank0():
        print(f"Loading data from {args.data_dir}...")
    
    # Windows 上禁用多进程，避免 pickle 问题
    import platform
    is_windows = platform.system() == 'Windows'
    if is_windows and args.num_workers > 0:
        if _is_rank0():
            print("Running on Windows, disabling num_workers to avoid pickle issues")
        args.num_workers = 0
        args.pin_memory = False
    
    if (args.dataset_mode or "multimodal").lower() == "text":
        from data.text_data_loader import build_text_dataloader

        train_loader = build_text_dataloader(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            packing=bool(args.packing),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            distributed=args.distributed,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
        )
    else:
        train_loader = get_data_loader(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            image_size=args.image_size,
            padding_mode=args.padding_mode,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            distributed=args.distributed,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
        )
    if _is_rank0():
        print(f"Data loader created. Number of batches: {len(train_loader)}")
    
    # 初始化模型
    if _is_rank0():
        print("Initializing model...")
    # 通过命令行参数切换配置（跨平台：Windows/macOS/Linux 一致）
    # --config_preset=local  -> configs/model_config_local.py:LocalModelConfig
    # --config_preset=prod7b -> configs/model_config_prod_7b.py:Prod7BModelConfig
    model_cfg = (args.config_preset or "default").strip().lower()
    if model_cfg == "local":
        from configs.model_config_local import LocalModelConfig
        config = LocalModelConfig()
        if _is_rank0():
            print("[config] Using LocalModelConfig (for local debug)")
    elif model_cfg == "prod7b":
        from configs.model_config_prod_7b import Prod7BModelConfig
        config = Prod7BModelConfig()
        if _is_rank0():
            print("[config] Using Prod7BModelConfig (7B/0.6B target, per-layer MoE)")
    else:
        config = ModelConfig()
        if _is_rank0():
            print("[config] Using default ModelConfig")

    # 允许 CLI 覆盖 attention backend（本地没装 flash-attn 也能跑）
    if args.attention_backend:
        config.attention_backend = args.attention_backend

    # -------------------------
    # P0-1：MoE backend 选择（环境变量覆盖，避免改代码）
    # -------------------------
    # 例：
    #   MOE_BACKEND=deepspeed torchrun --nproc_per_node=8 train/train_multimodal.py ...
    moe_backend_env = os.environ.get("MOE_BACKEND", "").strip()
    if moe_backend_env:
        config.moe_backend = moe_backend_env
        if _is_rank0():
            print(f"[config] moe_backend <- {config.moe_backend} (from env MOE_BACKEND)")

    # P0-2：Flash Attention 开关（环境变量覆盖）
    # 例：
    #   USE_FLASH_ATTN=1 torchrun ...
    use_flash_attn_env = os.environ.get("USE_FLASH_ATTN", "").strip()
    if use_flash_attn_env:
        config.use_flash_attn = use_flash_attn_env not in ("0", "false", "False", "no", "NO")
        if _is_rank0():
            print(f"[config] use_flash_attn <- {config.use_flash_attn} (from env USE_FLASH_ATTN)")

    use_multimodal = (args.dataset_mode or "multimodal").lower() != "text"
    model = HybridMMMoEModel(config, use_multimodal=use_multimodal)
    model.to(device)
    if _is_rank0():
        print("Model initialized and moved to device.")
    
    # 分布式训练包装（DDP）/ DeepSpeed
    engine = None
    if args.deepspeed:
        assert deepspeed is not None
        # 运行时生成 DS config（避免强依赖外部文件），并允许 --deepspeed_config 覆盖
        ds_config = None
        if args.deepspeed_config:
            ds_config = args.deepspeed_config
        else:
            ds_config = {
                "train_micro_batch_size_per_gpu": int(args.batch_size),
                "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
                "gradient_clipping": float(args.max_grad_norm),
                "zero_optimization": {
                    "stage": int(args.zero_stage),
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": 5e8,
                    "stage3_prefetch_bucket_size": 5e7,
                    "stage3_param_persistence_threshold": 1e6,
                },
                "bf16": {"enabled": bool(args.bf16)},
                "fp16": {"enabled": bool(args.fp16), "loss_scale": 0, "loss_scale_window": 1000},
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": float(args.learning_rate),
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": float(args.weight_decay),
                    },
                },
            }
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        model = engine
    elif args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    
    # -------------------------
    # 混合精度（AMP）
    # -------------------------
    # 约定：
    # - bf16 优先（更稳），fp16 需要 GradScaler
    if args.bf16 and args.fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled")
    use_amp = bool(args.bf16 or args.fp16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16) and device.type == "cuda")

    if not args.deepspeed:
        # 优化器配置
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        # 学习率调度器
        # total_steps 以 optimizer.step 次数为准（考虑梯度累积）
        grad_accum = max(1, int(args.gradient_accumulation_steps))
        total_steps = math.ceil((len(train_loader) * args.epochs) / grad_accum)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    else:
        # DeepSpeed 内部处理 optimizer/grad accumulation；scheduler 可后续再接入
        grad_accum = 1
        scheduler = None
    
    # -------------------------
    # 日志（rank0 结构化 jsonl）
    # -------------------------
    metrics_fp = None
    if _is_rank0():
        os.makedirs(args.output_dir, exist_ok=True)
        metrics_path = os.path.join(args.output_dir, "metrics_rank0.jsonl")
        metrics_fp = open(metrics_path, "a", encoding="utf-8")
        metrics_fp.write(json.dumps({"event": "start", "ts": datetime.utcnow().isoformat(), "args": vars(args)}) + "\n")
        metrics_fp.flush()

    # -------------------------
    # resume
    # -------------------------
    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    if args.resume_from and (not args.deepspeed):
        if _is_rank0():
            print(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = _load_checkpoint(
            path=args.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = ckpt.get("epoch", 0)
        start_step_in_epoch = ckpt.get("step_in_epoch", 0)
        global_step = ckpt.get("global_step", 0)
        if _is_rank0():
            print(f"Resumed from epoch {start_epoch}, step {start_step_in_epoch}, global step {global_step}")

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    model.train()
    stop_training = False
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            # DDP 下我们在 DataLoader 里使用 DistributedSampler
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
        
        start_time = time.time()
        total_loss = 0
        accum_loss = 0.0
        last_log_time = time.time()
        
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < start_step_in_epoch:
                continue

            data_t0 = time.time()
            # 移至设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch.get("pixel_values", None)
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            
            # 构造文本 positions（3D：[t,0,0]）
            B, T = input_ids.shape
            t = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            text_positions = torch.stack([t, torch.zeros_like(t), torch.zeros_like(t)], dim=-1)  # [B, T, 3]
            
            if use_multimodal:
                # Step 1：多模态序列对齐（关键）
                aligned = build_aligned_masks_and_labels(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_size=config.image_size,
                    patch_size=config.patch_size,
                    pad_ignore_index=-100,
                    image_pad_token_id=(getattr(config, "image_pad_token_id", None) if use_multimodal else None),
                    build_positions=True,
                )
                input_ids_total = aligned.input_ids_total
                attention_mask_total = aligned.attention_mask_total
                labels_total = aligned.labels_total
                positions_total = aligned.positions_total
            else:
                # text-only：不拼 image tokens，labels 仅对 padding 置 -100
                input_ids_total = input_ids
                attention_mask_total = attention_mask
                labels_total = input_ids.clone().masked_fill(attention_mask == 0, -100)
                positions_total = text_positions

            data_time = time.time() - data_t0
            
            # -------------------------
            # 前向传播
            # -------------------------
            # 启用 MTP 时我们需要 hidden_states（用于计算 shift=2..K 的 logits_step）
            need_hidden = bool(args.enable_mtp)

            # -------------------------
            # forward（AMP 可选）
            # -------------------------
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(
                    input_ids=input_ids_total,
                    positions=positions_total if positions_total is not None else text_positions,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask_total,
                    use_cache=False,
                    output_hidden_states=need_hidden,
                    # 让模型知道：input_ids_total 的前 T_img 个位置是 <|image_pad|> 占位符
                    image_pad_token_id=(getattr(config, "image_pad_token_id", None) if use_multimodal else None),
                )

            if need_hidden:
                logits, hidden_states, past_states, aux_loss = out
            else:
                logits, past_states, aux_loss = out
                hidden_states = None
            
            # 计算主任务损失
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels_total[:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                main_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            
            # 计算总损失：main loss + （可选）MTP loss + aux loss
            aux_loss_weight = 0.01
            loss = main_loss

            # -------------------------
            # MTP loss（shift=2..K），默认关闭
            # -------------------------
            if args.enable_mtp:
                base_model = model.module if hasattr(model, "module") else model
                mtp_out = mtp_loss_from_hidden(
                    hidden_states=hidden_states,
                    labels=labels_total,
                    lm_head=base_model.lm_head,
                    mtp_k=args.mtp_k,
                    ignore_index=-100,
                )
                loss = loss + float(args.mtp_weight) * mtp_out.loss_mtp
            else:
                mtp_out = type('obj', (object,), {'loss_mtp': torch.tensor(0.0)})()

            # MoE aux loss（负载均衡）
            loss = loss + aux_loss_weight * aux_loss
            
            # -------------------------
            # backward / step（DDP or DeepSpeed）
            # -------------------------
            if args.deepspeed:
                model.backward(loss)
                model.step()
                global_step = int(getattr(model, "global_steps", global_step + 1))
                grad_norm = 0.0
                step_time = time.time() - data_t0

                # DeepSpeed 分支也支持 max_steps（用于 smoke test / 调试）
                if args.max_steps and global_step >= int(args.max_steps):
                    stop_training = True
            else:
                loss_to_backward = loss / float(grad_accum)
                if scaler.is_enabled():
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

                do_step = ((step + 1) % grad_accum == 0) or (step + 1 == len(train_loader))
                if do_step:
                    step_t0 = time.time()
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).detach().cpu())

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    step_time = time.time() - step_t0

                # 估算吞吐：用有效 token 数（attention_mask_total=1 的位置）近似
                tokens_local = int(attention_mask_total.sum().item())
                tokens_total = tokens_local
                if args.distributed:
                    t_tensor = torch.tensor([tokens_local], device=device, dtype=torch.long)
                    torch.distributed.all_reduce(t_tensor, op=torch.distributed.ReduceOp.SUM)
                    tokens_total = int(t_tensor.item())
                tokens_per_sec = tokens_total / max(1e-6, step_time)

                # 分项 loss（用于日志）
                loss_main_v = float(main_loss.detach().cpu())
                loss_mtp_v = float(mtp_out.loss_mtp.detach().cpu()) if args.enable_mtp else 0.0
                aux_loss_v = float(aux_loss.detach().cpu()) if hasattr(aux_loss, "detach") else float(aux_loss)
                total_loss_v = float(loss.detach().cpu())

                if _is_rank0() and (global_step % args.log_interval == 0):
                    lr = float(optimizer.param_groups[0]["lr"]) if not args.deepspeed else float(model.get_lr()[0])
                    rec = {
                        "event": "train_step",
                        "ts": datetime.utcnow().isoformat(),
                        "epoch": int(epoch),
                        "step_in_epoch": int(step),
                        "global_step": int(global_step),
                        "lr": lr,
                        "loss": total_loss_v,
                        "loss_main": loss_main_v,
                        "loss_mtp": loss_mtp_v,
                        "aux_loss": aux_loss_v,
                        "grad_norm": grad_norm,
                        "tokens_total": int(tokens_total),
                        "tokens_per_sec": float(tokens_per_sec),
                        "data_time": float(data_time),
                        "step_time": float(step_time),
                    }
                    if metrics_fp is not None:
                        metrics_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        metrics_fp.flush()
                    print(
                        f"[gs={global_step}] loss={total_loss_v:.4f} main={loss_main_v:.4f} "
                        f"mtp={loss_mtp_v:.4f} aux={aux_loss_v:.4f} "
                        f"tok/s={tokens_per_sec:.1f} grad_norm={grad_norm:.2f}"
                    )

                # step-based checkpoint（生产常用：按步保存）
                if args.save_steps and (global_step % int(args.save_steps) == 0):
                    if args.deepspeed:
                        tag = f"global_step_{global_step}"
                        # DeepSpeed 保存需要所有 rank 都调用
                        model.save_checkpoint(args.output_dir, tag=tag)
                        if _is_rank0():
                            print(f"[deepspeed] Checkpoint saved to {args.output_dir} (tag={tag})")
                    else:
                        if _is_rank0():
                            ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step}.pt")
                            _save_checkpoint(
                                path=ckpt_path,
                                epoch=epoch,
                                step_in_epoch=step,
                                global_step=global_step,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                args=args,
                                config=config,
                            )
                            print(f"Checkpoint saved to {ckpt_path}")

                    # max_steps：用于 smoke test / 调试
                    if args.max_steps and global_step >= int(args.max_steps):
                        stop_training = True
            
            total_loss += loss.item()

            if stop_training:
                break
            
            # 打印日志
            if _is_rank0() and step % args.log_interval == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1}/{args.epochs}, Step {step}/{len(train_loader)}, Loss: {avg_loss:.4f}")
        
        if stop_training:
            if _is_rank0():
                print(f"[stop] reached max_steps={args.max_steps}, stop training.")
            break

        # 计算 epoch 时间
        epoch_time = time.time() - start_time
        if _is_rank0():
            avg_epoch_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            if args.deepspeed:
                tag = f"epoch_{epoch+1}"
                model.save_checkpoint(args.output_dir, tag=tag)
                if _is_rank0():
                    print(f"[deepspeed] Checkpoint saved to {args.output_dir} (tag={tag})")
            else:
                if _is_rank0():
                    checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
                    _save_checkpoint(
                        path=checkpoint_path,
                        epoch=epoch + 1,
                        step_in_epoch=0,
                        global_step=global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        args=args,
                        config=config,
                    )
                    print(f"Checkpoint saved to {checkpoint_path}")
    
    # 销毁进程组
    if args.distributed and (not args.deepspeed):
        destroy_process_group()
    
    if _is_rank0():
        if metrics_fp is not None:
            metrics_fp.write(json.dumps({"event": "end", "ts": datetime.utcnow().isoformat()}) + "\n")
            metrics_fp.close()
        print("Training completed!")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Multimodal Model Training')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory (should contain image_cache/ etc.)')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizers/qwen3-0.6b', help='Tokenizer path')
    
    # 模型参数
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--dataset_mode', type=str, default="multimodal", choices=["multimodal", "text"], help='Dataset mode: multimodal or text-only')
    parser.add_argument('--padding_mode', type=str, default="dynamic", choices=["dynamic", "max_length"], help='Padding mode for multimodal dataloader')
    parser.add_argument('--packing', action='store_true', help='Enable sample packing (text-only mode)')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument(
        '--config_preset',
        type=str,
        default='default',
        choices=['default', 'local', 'prod7b'],
        help='Model config preset: default / local (small for debugging) / prod7b (7B target)'
    )
    parser.add_argument('--attention_backend', type=str, default='', choices=['', 'torch', 'flash', 'flash_varlen'],
                        help='Attention backend override: torch / flash (fallback to torch if unavailable)')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')

    # 混合精度
    parser.add_argument('--bf16', action='store_true', help='Enable BF16 autocast (recommended if supported)')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 autocast + GradScaler')
    
    # 分布式训练
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--deepspeed', action='store_true', help='Enable DeepSpeed engine (ZeRO)')
    parser.add_argument('--zero_stage', type=int, default=3, choices=[1, 2, 3], help='DeepSpeed ZeRO stage')
    parser.add_argument('--deepspeed_config', type=str, default='', help='Path to DeepSpeed config json (optional)')

    # -------------------------
    # MTP（Multi-Token Prediction）
    # -------------------------
    parser.add_argument('--enable_mtp', action='store_true', help='Enable MTP (multi-token prediction) loss')
    parser.add_argument('--mtp_k', type=int, default=3, help='MTP prediction steps K (predict t+2..t+K)')
    parser.add_argument('--mtp_weight', type=float, default=0.3, help='Weight for MTP loss term')

    # DataLoader 参数
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--pin_memory', action='store_true', help='Enable pin_memory for DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data sampling')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=1, help='Save interval')
    parser.add_argument('--resume_from', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--save_steps', type=int, default=0, help='Save checkpoint every N optimizer steps (0=disable)')
    parser.add_argument('--max_steps', type=int, default=0, help='Stop training after N optimizer steps (0=disable)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 启动训练
    train(args)


if __name__ == "__main__":
    main()
