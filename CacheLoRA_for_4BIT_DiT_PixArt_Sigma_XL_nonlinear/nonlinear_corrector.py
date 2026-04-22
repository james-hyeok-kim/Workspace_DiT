"""
nonlinear_corrector.py

Nonlinear Cache-LoRA correctors for DeepCache.
4 options with increasing expressiveness:

  Option 1 (gelu):  correction = B @ GELU(A @ dx)
  Option 2 (mlp):   correction = W2 @ GELU(W1 @ dx + b1) + b2
  Option 3 (res):   h = GELU(W1 @ dx); h = h + GELU(W2 @ h); correction = W3 @ h
  Option 4 (film):  scale,shift = f(t); correction = W2 @ (scale * GELU(W1 @ dx) + shift)

Calibration: collect raw (dx, drift) pairs, then train with Adam + MSE.
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


def create_corrector(option: str, hidden_dim: int, rank: int = 4, mid_dim: int = 32):
    """Factory for corrector modules."""
    if option == "gelu":
        return GELUBottleneck(hidden_dim, rank)
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
    loss_type: str = "drift",        # "drift" | "fd" | "fd_weighted"
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
        (corrector_module, calib_time_sec)
    """
    import os

    # ---- Load cached corrector if available ----------------------------------
    if save_path and os.path.exists(save_path):
        print(f"  [NL-Corrector] Loading cached weights from {save_path}")
        t0 = time.perf_counter()

        # Determine hidden_dim to reconstruct model
        hidden_dim = None
        for name, param in transformer.transformer_blocks[cache_start].named_parameters():
            if param.ndim >= 2:
                hidden_dim = param.shape[-1]
                break

        model = create_corrector(corrector_type, hidden_dim, rank=rank, mid_dim=mid_dim)
        ckpt = torch.load(save_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model = model.half().to(device)
        model.eval()
        elapsed = time.perf_counter() - t0
        print(f"  [NL-Corrector] Loaded in {elapsed:.1f}s")
        return model, 0.0

    t0 = time.perf_counter()
    num_calib = min(num_calib, len(prompts))
    needs_t = (corrector_type == "film")

    print(f"  [NL-Corrector Calib] type={corrector_type}, {num_calib} prompts × "
          f"{t_count} steps, rank={rank}, mid={mid_dim}")

    # Determine hidden_dim
    hidden_dim = None
    for name, param in transformer.transformer_blocks[cache_start].named_parameters():
        if param.ndim >= 2:
            hidden_dim = param.shape[-1]
            break
    if hidden_dim is None:
        raise RuntimeError("Cannot determine hidden_dim")

    use_fd = loss_type in ("fd", "fd_weighted")

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

    # ---- Create and train corrector ------------------------------------------
    model = create_corrector(corrector_type, hidden_dim, rank=rank, mid_dim=mid_dim)
    model = model.to(device).float()

    # Move training data to GPU in batches
    batch_size = min(8192, N)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)

    dataset = torch.utils.data.TensorDataset(
        dx_all, target_all,
        *([] if t_all is None else [t_all]),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=0,
    )

    print(f"  [NL-Corrector Train] {train_epochs} epochs, lr={train_lr}, "
          f"batch={batch_size}, params={sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for epoch in range(train_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            if needs_t:
                bx, bd, bt = batch
                bx, bd, bt = bx.to(device), bd.to(device), bt.to(device)
                pred = model(bx, bt)
            else:
                bx, bd = batch
                bx, bd = bx.to(device), bd.to(device)
                pred = model(bx)

            if loss_type == "fd_weighted":
                with torch.no_grad():
                    token_var = bd.var(dim=-1, keepdim=True)
                    weight = token_var / (token_var.mean() + 1e-8)
                loss = (weight * (pred - bd).pow(2)).mean()
            else:
                loss = F.mse_loss(pred, bd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg = epoch_loss / max(n_batches, 1)
            print(f"    epoch {epoch+1:3d}/{train_epochs}  loss={avg:.6f}", flush=True)

    model.eval()

    # Compute final MSE and compare to linear baseline
    with torch.no_grad():
        eval_n = min(N, 32768)
        eval_dx = dx_all[:eval_n].to(device)
        eval_target = target_all[:eval_n].to(device)
        eval_t = t_all[:eval_n].to(device) if needs_t else None

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

    return model, calib_time
