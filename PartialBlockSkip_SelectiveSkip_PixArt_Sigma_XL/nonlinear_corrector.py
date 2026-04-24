"""
nonlinear_corrector.py

Nonlinear Cache-LoRA correctors for DeepCache.
4 options with increasing expressiveness:

  Option 1 (gelu):   correction = B @ GELU(A @ dx)
  Option 2 (mlp):    correction = W2 @ GELU(W1 @ dx + b1) + b2
  Option 3 (res):    h = GELU(W1 @ dx); h = h + GELU(W2 @ h); correction = W3 @ h
  Option 4 (film):   scale,shift = f(t); correction = W2 @ (scale * GELU(W1 @ dx) + shift)
  Option 5 (gelu_t): GELU bottleneck + FiLM-style scale-shift at bottleneck (Level 2)

Loss types:
  drift        — target = fresh_res - stale_res  (default)
  fd           — target = fresh_res = blocks(h_in) - h_in
  fd_weighted  — fd + per-token variance weighting
  fd_stratified — fd + timestep-bucket-equalized loss (Level 2)

Calibration: collect raw (dx, drift/target) pairs, then train with Adam.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Corrector modules
# ---------------------------------------------------------------------------

class GELUBottleneck(nn.Module):
    """Option 1: rank-k bottleneck + GELU. Same param count as linear."""
    def __init__(self, hidden_dim: int, rank: int):
        super().__init__()
        self.A = nn.Linear(hidden_dim, rank, bias=False)
        self.B = nn.Linear(rank, hidden_dim, bias=False)

    def forward(self, dx, t_norm=None):
        return self.B(F.gelu(self.A(dx)))


class BottleneckMLP(nn.Module):
    """Option 2: bottleneck MLP with bias."""
    def __init__(self, hidden_dim: int, mid_dim: int = 32):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, mid_dim)
        self.W2 = nn.Linear(mid_dim, hidden_dim)

    def forward(self, dx, t_norm=None):
        return self.W2(F.gelu(self.W1(dx)))


class ResidualMLP(nn.Module):
    """Option 3: 2-layer MLP with residual connection."""
    def __init__(self, hidden_dim: int, mid_dim: int = 32):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, mid_dim)
        self.W2 = nn.Linear(mid_dim, mid_dim)
        self.W3 = nn.Linear(mid_dim, hidden_dim)

    def forward(self, dx, t_norm=None):
        h = F.gelu(self.W1(dx))
        h = h + F.gelu(self.W2(h))
        return self.W3(h)


class FiLMCorrector(nn.Module):
    """Option 4: FiLM conditioning on normalized timestep."""
    def __init__(self, hidden_dim: int, mid_dim: int = 32):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, mid_dim)
        self.W2 = nn.Linear(mid_dim, hidden_dim)
        self.scale_net = nn.Linear(1, mid_dim)
        self.shift_net = nn.Linear(1, mid_dim)

    def forward(self, dx, t_norm):
        """t_norm: [B, T, 1] or broadcastable — normalized timestep in [0, 1]."""
        h = self.W1(dx)
        scale = self.scale_net(t_norm)
        shift = self.shift_net(t_norm)
        h = scale * F.gelu(h) + shift
        return self.W2(h)


class GELUBottleneckT(nn.Module):
    """Option 5 (Level 2): GELU bottleneck + FiLM-style timestep scale-shift at bottleneck.
    Adds ~2*rank params vs GELUBottleneck. Always requires t_norm."""
    def __init__(self, hidden_dim: int, rank: int):
        super().__init__()
        self.A = nn.Linear(hidden_dim, rank, bias=False)
        self.B = nn.Linear(rank, hidden_dim, bias=False)
        self.scale_net = nn.Linear(1, rank)
        self.shift_net  = nn.Linear(1, rank)

    def forward(self, dx, t_norm=None):
        h = self.A(dx)                              # [N, rank]
        if t_norm is not None:
            scale = self.scale_net(t_norm)          # [N, rank]
            shift = self.shift_net(t_norm)          # [N, rank]
            h = scale * F.gelu(h) + shift
        else:
            h = F.gelu(h)
        return self.B(h)


def create_corrector(option: str, hidden_dim: int, rank: int = 4, mid_dim: int = 32):
    """Factory for corrector modules."""
    if option == "gelu":
        return GELUBottleneck(hidden_dim, rank)
    elif option == "gelu_t":
        return GELUBottleneckT(hidden_dim, rank)
    elif option == "mlp":
        return BottleneckMLP(hidden_dim, mid_dim)
    elif option == "res":
        return ResidualMLP(hidden_dim, mid_dim)
    elif option == "film":
        return FiLMCorrector(hidden_dim, mid_dim)
    else:
        raise ValueError(f"Unknown corrector option: {option}")


# ---------------------------------------------------------------------------
# Calibration: collect (dx, drift) pairs and train
# ---------------------------------------------------------------------------

def calibrate_nonlinear_corrector(
    pipe,
    transformer,
    cache_start: int,
    cache_end: int,
    cache_interval: int,
    prompts: list,
    num_calib: int,
    t_count: int,
    guidance_scale: float,
    device,
    corrector_type: str = "gelu",   # gelu / mlp / res / film
    rank: int = 4,
    mid_dim: int = 32,
    train_epochs: int = 300,
    train_lr: float = 1e-3,
    calib_seed_offset: int = 1000,
    save_path: str = None,           # 학습된 weights 저장/로드 경로
    loss_type: str = "drift",        # "drift" | "fd" | "fd_weighted" | "fd_stratified"
    n_buckets: int = 5,              # timestep buckets for fd_stratified
    return_data: bool = False,       # if True, return (model, time, dx_gpu, target_gpu)
):
    """
    Collect calibration pairs, then train a nonlinear corrector.

    loss_type:
      "drift"       — target = (fresh_res - stale_res).  Current default.
      "fd"          — target = fresh_res = blocks(h_in) - h_in.  Stronger signal.
      "fd_weighted" — same as "fd" but loss weighted by per-token variance of target.

    If save_path exists: load and return immediately (skip calib+training).
    If save_path provided after training: save weights for future reuse.

    Returns:
        (corrector_module, calib_time_sec)               if return_data=False
        (corrector_module, calib_time_sec, dx_gpu, tgt)  if return_data=True
    """
    import os

    # Determine hidden_dim (shared by both cache-load and training paths)
    hidden_dim = None
    for name, param in transformer.transformer_blocks[cache_start].named_parameters():
        if param.ndim >= 2:
            hidden_dim = param.shape[-1]
            break
    if hidden_dim is None:
        raise RuntimeError("Cannot determine hidden_dim")

    model_preloaded = None

    # ---- Load cached corrector if available ----------------------------------
    if save_path and os.path.exists(save_path):
        print(f"  [NL-Corrector] Loading cached weights from {save_path}")
        t0_load = time.perf_counter()
        model_preloaded = create_corrector(corrector_type, hidden_dim, rank=rank, mid_dim=mid_dim)
        ckpt = torch.load(save_path, map_location="cpu")
        model_preloaded.load_state_dict(ckpt["state_dict"])
        model_preloaded = model_preloaded.half().to(device)
        model_preloaded.eval()
        elapsed = time.perf_counter() - t0_load
        print(f"  [NL-Corrector] Loaded in {elapsed:.1f}s")
        if not return_data:
            return model_preloaded, 0.0
        print(f"  [NL-Corrector] Collecting drift data (return_data=True)...")

    t0 = time.perf_counter()
    num_calib = min(num_calib, len(prompts))

    print(f"  [NL-Corrector Calib] type={corrector_type}, {num_calib} prompts × "
          f"{t_count} steps, rank={rank}, mid={mid_dim}")

    use_fd = loss_type in ("fd", "fd_weighted", "fd_stratified")
    # t_norm needed: film and gelu_t always; stratified loss also needs t for bucketing
    needs_t = (corrector_type in ("film", "gelu_t")) or (loss_type == "fd_stratified")

    # ---- Hook-based collection -----------------------------------------------
    step_counter = [0]
    h_in_buf = {}
    h_out_buf = {}
    _tmp_in = [None]

    def _pre_hook(module, args):
        _tmp_in[0] = args[0].detach().cpu()

    def _post_hook(module, args, output):
        if _tmp_in[0] is not None:
            s = step_counter[0]
            h_in_buf[s] = _tmp_in[0]
            h_out_buf[s] = output.detach().cpu()
            _tmp_in[0] = None
            step_counter[0] += 1

    h_pre = transformer.transformer_blocks[cache_start].register_forward_pre_hook(_pre_hook)
    h_post = transformer.transformer_blocks[cache_end - 1].register_forward_hook(_post_hook)

    # Collect calibration pairs (dx, target, t_norm)
    # drift mode:  target = fresh_res_curr - stale_res_prev  (residual of residual)
    # fd mode:     target = fresh_res_curr = h_out_c - h_in_c  (direct block output delta)
    all_dx = []
    all_target = []
    all_t = []

    try:
        for i in range(num_calib):
            step_counter[0] = 0
            h_in_buf.clear()
            h_out_buf.clear()
            _tmp_in[0] = None

            gen = torch.Generator(device=device).manual_seed(calib_seed_offset + i)
            with torch.no_grad():
                pipe(
                    prompts[i % len(prompts)],
                    num_inference_steps=t_count,
                    guidance_scale=guidance_scale,
                    generator=gen,
                )

            n_collected = step_counter[0]
            for s in range(cache_interval, n_collected):
                s_prev = s - cache_interval
                if s_prev not in h_in_buf:
                    continue
                h_in_c = h_in_buf[s].float()
                h_in_p = h_in_buf[s_prev].float()
                h_out_c = h_out_buf[s].float()
                h_out_p = h_out_buf[s_prev].float()

                dx = (h_in_c - h_in_p).reshape(-1, hidden_dim)
                if use_fd:
                    # Feature distillation: target = full block output delta (stronger signal)
                    target = (h_out_c - h_in_c).reshape(-1, hidden_dim)
                else:
                    # Original drift: target = change in residual between steps
                    target = ((h_out_c - h_in_c) - (h_out_p - h_in_p)).reshape(-1, hidden_dim)

                all_dx.append(dx)
                all_target.append(target)
                if needs_t:
                    t_norm_val = s / max(n_collected - 1, 1)
                    all_t.append(torch.full((dx.shape[0], 1), t_norm_val))

            h_in_buf.clear()
            h_out_buf.clear()
            print(f"    calib {i+1}/{num_calib} done", flush=True)
    finally:
        h_pre.remove()
        h_post.remove()

    dx_all = torch.cat(all_dx, dim=0)         # [N, H]
    target_all = torch.cat(all_target, dim=0)  # [N, H]
    t_all = torch.cat(all_t, dim=0) if needs_t else None  # [N, 1] or None
    N = dx_all.shape[0]
    print(f"  [NL-Corrector Calib] Collected {N:,} samples, H={hidden_dim}, "
          f"loss_type={loss_type}")

    # Move tensors to GPU (needed for both early-exit and training paths)
    dx_gpu     = dx_all.to(device)
    target_gpu = target_all.to(device)
    t_gpu      = t_all.to(device) if t_all is not None else None

    # If model was pre-loaded from cache, skip training and return with data
    if model_preloaded is not None:
        calib_time = time.perf_counter() - t0
        print(f"  [NL-Corrector] Data collected in {calib_time:.1f}s (model from cache)")
        return (model_preloaded, calib_time, dx_gpu, target_gpu) if return_data else (model_preloaded, calib_time)

    # ---- Create and train corrector ------------------------------------------
    model = create_corrector(corrector_type, hidden_dim, rank=rank, mid_dim=mid_dim)
    model = model.to(device).float()

    batch_size = min(8192, N)
    n_batches_per_epoch = (N + batch_size - 1) // batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)

    print(f"  [NL-Corrector Train] {train_epochs} epochs, lr={train_lr}, "
          f"batch={batch_size}, params={sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for epoch in range(train_epochs):
        epoch_loss = 0.0
        perm = torch.randperm(N, device=device)
        for i in range(n_batches_per_epoch):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            bx = dx_gpu[idx]
            bd = target_gpu[idx]
            bt = t_gpu[idx] if t_gpu is not None else None
            pred = model(bx, bt) if needs_t else model(bx)

            if loss_type == "fd_weighted":
                with torch.no_grad():
                    token_var = bd.var(dim=-1, keepdim=True)
                    weight = token_var / (token_var.mean() + 1e-8)
                loss = (weight * (pred - bd).pow(2)).mean()
            elif loss_type == "fd_stratified":
                per_sample_loss = (pred - bd).pow(2).mean(dim=-1)  # [B]
                bucket = (bt.squeeze(-1) * n_buckets).long().clamp(0, n_buckets - 1)
                bucket_losses = []
                for b in range(n_buckets):
                    mask = bucket == b
                    if mask.any():
                        bucket_losses.append(per_sample_loss[mask].mean())
                loss = torch.stack(bucket_losses).mean()
            else:
                loss = F.mse_loss(pred, bd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg = epoch_loss / max(n_batches_per_epoch, 1)
            print(f"    epoch {epoch+1:3d}/{train_epochs}  loss={avg:.6f}", flush=True)

    model.eval()

    # Compute final MSE and compare to linear baseline
    with torch.no_grad():
        eval_n = min(N, 32768)
        eval_dx     = dx_gpu[:eval_n]
        eval_target = target_gpu[:eval_n]
        eval_t      = t_gpu[:eval_n] if t_gpu is not None else None

        if needs_t:
            pred = model(eval_dx, eval_t)
        else:
            pred = model(eval_dx)
        nl_mse = F.mse_loss(pred, eval_target).item()

        # Linear baseline MSE (from SVD) — always on same target for fair comparison
        C = eval_target.double().T @ eval_dx.double()
        C_norm = (C / eval_n).float()
        U, S, Vt = torch.linalg.svd(C_norm, full_matrices=False)
        sq = S[:rank].clamp(min=0.0).sqrt()
        A_lin = (sq.unsqueeze(1) * Vt[:rank, :])
        B_lin = (U[:, :rank] * sq.unsqueeze(0))
        lin_pred = F.linear(F.linear(eval_dx, A_lin), B_lin)
        lin_mse = F.mse_loss(lin_pred, eval_target).item()

    print(f"  [NL-Corrector] MSE ({loss_type}):  "
          f"linear={lin_mse:.6f}  nonlinear={nl_mse:.6f}  "
          f"ratio={nl_mse/max(lin_mse, 1e-10):.4f}")

    calib_time = time.perf_counter() - t0
    print(f"  [NL-Corrector] Done in {calib_time:.1f}s")

    model = model.half()  # save VRAM

    # ---- Save trained weights for future reuse --------------------------------
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "corrector_type": corrector_type,
            "hidden_dim": hidden_dim,
            "rank": rank,
            "mid_dim": mid_dim,
            "t_count": t_count,
            "cache_start": cache_start,
            "cache_end": cache_end,
            "loss_type": loss_type,
            "state_dict": model.cpu().state_dict(),
        }, save_path)
        model = model.to(device)
        print(f"  [NL-Corrector] Weights saved to {save_path}")

    return (model, calib_time, dx_gpu, target_gpu) if return_data else (model, calib_time)


# ---------------------------------------------------------------------------
# Level 3: Trajectory Distillation fine-tuning
# ---------------------------------------------------------------------------

def _traj_denoise_step(transformer, latents, t, prompt_embeds_cfg, attn_mask_cfg,
                        guidance_scale, scheduler):
    """One denoising step (manual CFG). Returns new latents (keeps grad if enabled)."""
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    current_timestep = t.float().expand(latent_model_input.shape[0])

    noise_pred = transformer(
        latent_model_input,
        encoder_hidden_states=prompt_embeds_cfg,
        encoder_attention_mask=attn_mask_cfg,
        timestep=current_timestep,
        added_cond_kwargs={"resolution": None, "aspect_ratio": None},
        return_dict=False,
    )[0]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Handle learned sigma (out_channels == 8)
    if transformer.config.out_channels // 2 == latents.shape[1]:
        noise_pred = noise_pred.chunk(2, dim=1)[0]

    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents


def fine_tune_with_trajectory_distillation(
    nl_corrector,
    cache_state,
    fp16_pipe,
    student_pipe,
    prompts: list,
    num_calib: int,
    t_count: int,
    guidance_scale: float,
    device,
    K: int = 6,
    lambda_traj: float = 0.1,
    n_iter: int = 200,
    lr: float = 5e-5,
    calib_seed_offset: int = 2000,
    save_path: str = None,
):
    """
    Fine-tune nl_corrector in-place using K-step trajectory distillation.

    Teacher: fp16_pipe (full model, no_grad).
    Student: student_pipe (SVDQUANT + DeepCache + corrector).
    Loss: MSE between student and teacher latents after K denoising steps.

    Returns: fine-tuned nl_corrector (same object, updated in-place).
    """
    import copy
    import random
    import os

    num_calib = min(num_calib, len(prompts))
    latent_h = latent_w = student_pipe.transformer.config.sample_size  # 128 for 1024px

    # ── 1. Pre-encode prompts ─────────────────────────────────────────────────
    print(f"  [TrajDistill] Encoding {num_calib} prompts...")
    all_prompt_embeds = []
    all_attn_masks = []
    with torch.no_grad():
        for i in range(num_calib):
            pe, pa, ne, na = fp16_pipe.encode_prompt(
                prompt=prompts[i % len(prompts)],
                do_classifier_free_guidance=True,
                device=device,
            )
            all_prompt_embeds.append(torch.cat([ne, pe]))   # [2, seq, D]
            all_attn_masks.append(torch.cat([na, pa]))      # [2, seq]

    # ── 2. Pre-run teacher trajectories ──────────────────────────────────────
    print(f"  [TrajDistill] Collecting teacher trajectories ({num_calib} prompts × {t_count} steps)...")
    teacher_sched = copy.deepcopy(fp16_pipe.scheduler)
    teacher_sched.set_timesteps(t_count, device=device)
    timesteps = teacher_sched.timesteps

    all_teacher_traj = []  # list of lists: [t_count+1 tensors] per prompt
    with torch.no_grad():
        for i in range(num_calib):
            gen = torch.Generator(device=device).manual_seed(calib_seed_offset + i)
            x = torch.randn(1, 4, latent_h, latent_w, generator=gen,
                            device=device, dtype=torch.float16)
            x = x * teacher_sched.init_noise_sigma
            traj = [x.clone()]
            sched_i = copy.deepcopy(teacher_sched)
            for t in timesteps:
                x = _traj_denoise_step(fp16_pipe.transformer, x, t,
                                       all_prompt_embeds[i], all_attn_masks[i],
                                       guidance_scale, sched_i)
                traj.append(x.detach().clone())
            all_teacher_traj.append(traj)
            print(f"    teacher traj {i+1}/{num_calib} done", flush=True)

    # ── 3. Fine-tune corrector ────────────────────────────────────────────────
    nl_corrector.float().train()
    for p in nl_corrector.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(nl_corrector.parameters(), lr=lr)

    student_transformer = student_pipe.transformer
    loss_history = []

    print(f"  [TrajDistill] Fine-tuning: {n_iter} iters, K={K}, λ_traj={lambda_traj}, lr={lr}")
    for it in range(n_iter):
        prompt_idx = random.randint(0, num_calib - 1)
        start_step = random.randint(0, max(0, t_count - K - 1))

        # Initial noise (same as teacher)
        gen = torch.Generator(device=device).manual_seed(calib_seed_offset + prompt_idx)
        x = torch.randn(1, 4, latent_h, latent_w, generator=gen,
                        device=device, dtype=torch.float16)

        student_sched = copy.deepcopy(fp16_pipe.scheduler)
        student_sched.set_timesteps(t_count, device=device)
        x = x * student_sched.init_noise_sigma

        # Warm-up: run student steps 0..start_step without gradient
        cache_state.reset()
        cache_state.nl_train_mode = False
        with torch.no_grad():
            for si in range(start_step):
                t = student_sched.timesteps[si]
                x = _traj_denoise_step(student_transformer, x, t,
                                       all_prompt_embeds[prompt_idx],
                                       all_attn_masks[prompt_idx],
                                       guidance_scale, student_sched)

        # K gradient steps
        cache_state.nl_train_mode = True
        x = x.detach()
        for si in range(start_step, min(start_step + K, t_count)):
            t = student_sched.timesteps[si]
            x = _traj_denoise_step(student_transformer, x, t,
                                   all_prompt_embeds[prompt_idx],
                                   all_attn_masks[prompt_idx],
                                   guidance_scale, student_sched)

        # Trajectory loss vs teacher
        x_teacher = all_teacher_traj[prompt_idx][min(start_step + K, t_count)].to(device)
        traj_loss = F.mse_loss(x.float(), x_teacher.float())
        loss = lambda_traj * traj_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nl_corrector.parameters(), 1.0)
        optimizer.step()
        loss_history.append(traj_loss.item())

        if (it + 1) % 50 == 0:
            avg = sum(loss_history[-50:]) / 50
            print(f"  [TrajDistill] iter {it+1}/{n_iter}  traj_loss={avg:.4f}", flush=True)

    # Reset to inference mode
    cache_state.nl_train_mode = False
    nl_corrector.half().eval()
    for p in nl_corrector.parameters():
        p.requires_grad_(False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state_dict": nl_corrector.state_dict(), "loss_type": "traj_distill"}, save_path)
        print(f"  [TrajDistill] Saved to {save_path}")

    print(f"  [TrajDistill] Done. Final avg traj_loss={sum(loss_history[-20:])/20:.4f}")
    return nl_corrector


# ---------------------------------------------------------------------------
# Level 3 (combined): Feature loss + Trajectory Distillation
# ---------------------------------------------------------------------------

def fine_tune_combined(
    nl_corrector,
    cache_state,
    fp16_pipe,
    student_pipe,
    drift_data,           # (dx_gpu, target_gpu) — GPU tensors from calibrate_nonlinear_corrector(return_data=True)
    prompts: list,
    num_calib: int,
    t_count: int,
    guidance_scale: float,
    device,
    K: int = 6,
    lambda_traj: float = 0.1,
    lambda_warmup_frac: float = 0.2,
    n_epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 8192,
    calib_seed_offset: int = 2000,
    save_path: str = None,
):
    """
    Combined feature + trajectory distillation fine-tuning.

    Each iteration:
      feature_loss = MSE(corrector(dx_batch), target_batch)   — offline drift pairs
      traj_loss    = MSE(student_latents, teacher_latents)     — K-step online rollout
      total_loss   = feature_loss + lambda_traj * traj_loss

    lambda_traj linearly warms up from 0 → lambda_traj over first lambda_warmup_frac of iters.
    Teacher = fp16_pipe (no_grad). Student = student_pipe (SVDQUANT + DeepCache + corrector).
    Gradient flows only through nl_corrector (SVDQUANT weights frozen).

    Returns: fine-tuned nl_corrector (updated in-place).
    """
    import copy, random, os

    if save_path and os.path.exists(save_path):
        print(f"  [CombinedTraj] Loading cached combined weights from {save_path}")
        ckpt = torch.load(save_path, map_location="cpu")
        nl_corrector.load_state_dict(ckpt["state_dict"])
        nl_corrector = nl_corrector.half().to(device)
        nl_corrector.eval()
        for p in nl_corrector.parameters():
            p.requires_grad_(False)
        return nl_corrector

    num_calib = min(num_calib, len(prompts))
    latent_h = latent_w = student_pipe.transformer.config.sample_size
    dx_gpu, target_gpu = drift_data
    N = dx_gpu.shape[0]

    # ── 1. Encode prompts ─────────────────────────────────────────────────
    print(f"  [CombinedTraj] Encoding {num_calib} prompts...")
    all_prompt_embeds = []
    all_attn_masks = []
    with torch.no_grad():
        for i in range(num_calib):
            pe, pa, ne, na = fp16_pipe.encode_prompt(
                prompt=prompts[i % len(prompts)],
                do_classifier_free_guidance=True,
                device=device,
            )
            all_prompt_embeds.append(torch.cat([ne, pe]))
            all_attn_masks.append(torch.cat([na, pa]))

    # ── 2. Pre-compute teacher trajectories ──────────────────────────────
    print(f"  [CombinedTraj] Collecting teacher trajectories ({num_calib} × {t_count} steps)...")
    teacher_sched = copy.deepcopy(fp16_pipe.scheduler)
    teacher_sched.set_timesteps(t_count, device=device)
    timesteps = teacher_sched.timesteps

    all_teacher_traj = []
    with torch.no_grad():
        for i in range(num_calib):
            gen = torch.Generator(device=device).manual_seed(calib_seed_offset + i)
            x = torch.randn(1, 4, latent_h, latent_w, generator=gen,
                            device=device, dtype=torch.float16)
            x = x * teacher_sched.init_noise_sigma
            traj = [x.clone()]
            sched_i = copy.deepcopy(teacher_sched)
            for t in timesteps:
                x = _traj_denoise_step(fp16_pipe.transformer, x, t,
                                       all_prompt_embeds[i], all_attn_masks[i],
                                       guidance_scale, sched_i)
                traj.append(x.detach().clone())
            all_teacher_traj.append(traj)
            print(f"    teacher traj {i+1}/{num_calib} done", flush=True)

    # ── 3. Combined fine-tuning ──────────────────────────────────────────
    nl_corrector.float().train()
    for p in nl_corrector.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(nl_corrector.parameters(), lr=lr)

    student_transformer = student_pipe.transformer
    warmup_iters = max(1, int(n_epochs * lambda_warmup_frac))
    feat_losses = []
    traj_losses = []

    print(f"  [CombinedTraj] Training: {n_epochs} iters, K={K}, λ_traj={lambda_traj}, "
          f"warmup={warmup_iters} iters, lr={lr}")

    for it in range(n_epochs):
        lam = lambda_traj * min(1.0, it / warmup_iters)

        # Feature batch — random sample from pre-collected drift data
        idx = torch.randint(0, N, (min(batch_size, N),), device=device)
        bx = dx_gpu[idx]
        bd = target_gpu[idx]
        pred = nl_corrector(bx)
        feat_loss = F.mse_loss(pred, bd)

        # Trajectory rollout — K steps with gradient through corrector
        prompt_idx = random.randint(0, num_calib - 1)
        start_step = random.randint(0, max(0, t_count - K - 1))

        gen = torch.Generator(device=device).manual_seed(calib_seed_offset + prompt_idx)
        x = torch.randn(1, 4, latent_h, latent_w, generator=gen,
                        device=device, dtype=torch.float16)
        student_sched = copy.deepcopy(fp16_pipe.scheduler)
        student_sched.set_timesteps(t_count, device=device)
        x = x * student_sched.init_noise_sigma

        # Warm-up to start_step without gradient
        cache_state.reset()
        cache_state.nl_train_mode = False
        with torch.no_grad():
            for si in range(start_step):
                t = student_sched.timesteps[si]
                x = _traj_denoise_step(student_transformer, x, t,
                                       all_prompt_embeds[prompt_idx],
                                       all_attn_masks[prompt_idx],
                                       guidance_scale, student_sched)

        # K gradient steps — corrector receives gradient, SVDQUANT frozen
        cache_state.nl_train_mode = True
        x = x.detach()
        for si in range(start_step, min(start_step + K, t_count)):
            t = student_sched.timesteps[si]
            x = _traj_denoise_step(student_transformer, x, t,
                                   all_prompt_embeds[prompt_idx],
                                   all_attn_masks[prompt_idx],
                                   guidance_scale, student_sched)

        x_teacher = all_teacher_traj[prompt_idx][min(start_step + K, t_count)].to(device)
        traj_loss = F.mse_loss(x.float(), x_teacher.float())

        total_loss = feat_loss + lam * traj_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(nl_corrector.parameters(), 1.0)
        optimizer.step()

        feat_losses.append(feat_loss.item())
        traj_losses.append(traj_loss.item())

        if (it + 1) % 20 == 0 or it == 0:
            avg_feat = sum(feat_losses[-20:]) / min(20, len(feat_losses))
            avg_traj = sum(traj_losses[-20:]) / min(20, len(traj_losses))
            print(f"  [CombinedTraj] iter {it+1}/{n_epochs}  feat={avg_feat:.6f}  "
                  f"traj={avg_traj:.4f}  λ={lam:.4f}", flush=True)

    # Reset to inference mode
    cache_state.nl_train_mode = False
    nl_corrector.half().eval()
    for p in nl_corrector.parameters():
        p.requires_grad_(False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state_dict": nl_corrector.state_dict(), "loss_type": "drift_traj"}, save_path)
        print(f"  [CombinedTraj] Saved to {save_path}")

    avg_feat = sum(feat_losses[-20:]) / min(20, len(feat_losses))
    avg_traj = sum(traj_losses[-20:]) / min(20, len(traj_losses))
    print(f"  [CombinedTraj] Done. feat={avg_feat:.6f}  traj={avg_traj:.4f}")
    return nl_corrector
