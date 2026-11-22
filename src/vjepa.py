"""V-JEPA 2 helpers (transformers==4.57.1, VJEPA2Model).

Verified in transformers 4.57.1 source (models/vjepa2/modeling_vjepa2.py):
* VJEPA2Model.forward runs the encoder on ALL frames, and only afterwards uses
  context_mask/target_mask to pick predictor inputs. Using it for forecasting
  would leak future frames into the context through self-attention. We never
  call VJEPA2Model.forward; we call model.encoder on past frames only and then
  model.predictor directly.
* VJEPA2Encoder.forward(pixel_values_videos [B,T,3,H,W], head_mask=None,
  output_attentions=False, output_hidden_states=False) -> BaseModelOutput.
  hidden_states[i] = input of block i (i=0: patch embeddings); the last entry
  is the final LayerNorm output (== last_hidden_state). Token order is (t,h,w),
  tokens_per_slot = (crop/patch)^2, one temporal slot = 2 frames (tubelet).
* VJEPA2Predictor.forward(encoder_hidden_states, context_mask: list[LongTensor[B,n]],
  target_mask: list[LongTensor[B,m]], ...) -> BaseModelOutput with
  last_hidden_state [B, m, hidden_size]: gathered context tokens + learned mask
  tokens at the target indices, RoPE positions from the index values,
  12 layers, LayerNorm, keep target tokens, Linear back to hidden_size.
  Target indices may exceed the encoder sequence (mask tokens are repeated
  to target_mask[0].max()+1), and all batch rows must share the same masks.
* Training target (facebookresearch/vjepa2 app/vjepa/train.py forward_target):
  F.layer_norm(target_encoder(clip)) without affine params; loss = mean |z - h|.
  So forecasts are compared with layer_norm(encoder(full clip))[target tokens].
"""
import copy

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)   # VJEPA2 video processor normalisation
IMAGENET_STD = (0.229, 0.224, 0.225)


def torch_dtype(name):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_models(cfg, device):
    """Returns (model, rand_predictor, tokens_per_slot)."""
    from transformers import AutoModel, VJEPA2Config, VJEPA2Model
    dt = torch_dtype(cfg["model"]["dtype"])
    if cfg["model"].get("tiny_random"):
        conf = VJEPA2Config(crop_size=cfg["video"]["size"], patch_size=16, tubelet_size=2, frames_per_clip=16,
                            hidden_size=128, num_hidden_layers=10, num_attention_heads=2, mlp_ratio=2.0,
                            pred_hidden_size=64, pred_num_hidden_layers=2, pred_num_attention_heads=2)
        torch.manual_seed(0)
        model = VJEPA2Model(conf)
    else:
        model = AutoModel.from_pretrained(cfg["model"]["name"], attn_implementation="sdpa")
    model = model.to(device=device, dtype=dt).eval()
    rand = copy.deepcopy(model.predictor)
    torch.manual_seed(cfg["extract"]["seed_rand_predictor"])
    rand.apply(model._init_weights)          # model's own init: trunc_normal(0.02) Linear, LN=1/0, zero mask tokens
    rand = rand.to(device=device, dtype=dt).eval()
    assert not torch.equal(rand.proj.weight, model.predictor.proj.weight), "random predictor was not re-initialised"
    c = model.config
    tps = (c.crop_size // c.patch_size) ** 2
    assert c.tubelet_size == 2
    return model, rand, tps


class Normalizer:
    def __init__(self, device, dtype):
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
        self.dtype = dtype
        self.device = device

    def __call__(self, frames_u8):
        """frames_u8: numpy or torch uint8 [B,T,H,W,3] -> [B,T,3,H,W] normalised."""
        x = torch.as_tensor(frames_u8).to(self.device, non_blocking=True)
        x = x.permute(0, 1, 4, 2, 3).float().div_(255.0)
        return ((x - self.mean) / self.std).to(self.dtype)


def encode(model, x, mid_layer=None):
    out = model.encoder(pixel_values_videos=x, output_hidden_states=mid_layer is not None)
    mid = out.hidden_states[mid_layer] if mid_layer is not None else None
    return out.last_hidden_state, mid


def masks(B, n_ctx_slots, tgt_slots, tps, device):
    cm = torch.arange(0, n_ctx_slots * tps, device=device).unsqueeze(0).repeat(B, 1)
    tm = torch.cat([torch.arange(s * tps, (s + 1) * tps, device=device) for s in tgt_slots]).unsqueeze(0).repeat(B, 1)
    return [cm], [tm]


def predict(predictor, ctx_tokens, cm, tm):
    return predictor(encoder_hidden_states=ctx_tokens, context_mask=cm, target_mask=tm).last_hidden_state


def target_ln(x):
    """Non-affine layer norm used on V-JEPA 2 training targets."""
    x = x.float()
    return F.layer_norm(x, (x.shape[-1],))


def l1_tokens(a, b):
    """Mean |a-b| over tokens and dims, per batch row."""
    return (a.float() - b.float()).abs().mean(dim=(1, 2))


def cos_tokens(a, b):
    """Mean over tokens of cosine similarity, per batch row."""
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean(dim=1)


@torch.inference_mode()
def batch_features(cfg, model, rand_pred, tps, norm, Fr, E, return_tokens=False):
    """All features for one batch of TRs with frame ends E (list of e_k).
    Fr: uint8 [nF,H,W,3]. Returns dict of numpy arrays [B] or [B,D]."""
    ex = cfg["extract"]
    CTX, FUT, LONG = ex["ctx_frames"], ex["fut_frames"], ex["long_frames"]
    cs, fs = CTX // 2, FUT // 2
    B, dev = len(E), norm.device
    for e in E:
        assert e - LONG >= 0 and e - CTX >= 0 and e + FUT <= len(Fr)
    import numpy as np
    full = norm(np.stack([Fr[e - CTX:e + FUT] for e in E]))
    past = full[:, :CTX].contiguous()                      # only frames before t_k
    h, mid = encode(model, past, cfg["model"]["layer_mid"])
    assert h.shape[1] == cs * tps, (h.shape, cs, tps)
    T_raw, _ = encode(model, full)
    T_raw = T_raw[:, cs * tps:(cs + fs) * tps]             # encoder's view of the actual next 2 s
    Tn = target_ln(T_raw)
    cm, tm = masks(B, cs, range(cs, cs + fs), tps, dev)
    P = predict(model.predictor, h, cm, tm)
    Pr = predict(rand_pred, h, cm, tm)
    assert P.shape == (B, fs * tps, h.shape[-1]), P.shape
    Cp = target_ln(h[:, (cs - 1) * tps:cs * tps]).repeat(1, fs, 1)
    Lg, _ = encode(model, norm(np.stack([Fr[e - LONG:e] for e in E])))
    f = lambda t: t.float().cpu().numpy()
    out = {
        "c_mid": f(mid.float().mean(1)), "c_last": f(h.float().mean(1)), "clong": f(Lg.float().mean(1)),
        "hhat": f(P.float().mean(1)), "hhat_rand": f(Pr.float().mean(1)), "oracle": f(T_raw.float().mean(1)),
        "fore_l1": f(l1_tokens(P, Tn)), "fore_cos": f(cos_tokens(P, Tn)),
        "rand_l1": f(l1_tokens(Pr, Tn)), "rand_cos": f(cos_tokens(Pr, Tn)),
        "copy_l1": f(l1_tokens(Cp, Tn)), "copy_cos": f(cos_tokens(Cp, Tn)),
    }
    if return_tokens:
        out["_tokens"] = {"h": h, "P": P, "Pr": Pr, "Tn": Tn, "full": full}
    return out


def quadrant_pool(x, n_slots, grid):
    """x [B, n_slots*grid*grid, D] in (t,h,w) token order -> [B, 4, D]: mean over slots and each
    2x2 spatial quadrant (top-left, top-right, bottom-left, bottom-right)."""
    B, _, D = x.shape
    g2 = grid // 2
    x = x.float().reshape(B, n_slots, grid, grid, D)
    return torch.stack([x[:, :, :g2, :g2].mean((1, 2, 3)), x[:, :, :g2, g2:].mean((1, 2, 3)),
                        x[:, :, g2:, :g2].mean((1, 2, 3)), x[:, :, g2:, g2:].mean((1, 2, 3))], 1)


def quadrant_err(a, b, n_slots, grid):
    """Per-quadrant mean L1 and mean cosine between token sets a, b [B, n_slots*grid*grid, D] -> 2 x [B, 4]."""
    B, _, D = a.shape
    g2 = grid // 2
    l1 = (a.float() - b.float()).abs().mean(-1).reshape(B, n_slots, grid, grid)
    cs = F.cosine_similarity(a.float(), b.float(), dim=-1).reshape(B, n_slots, grid, grid)
    q = lambda t: torch.stack([t[:, :, :g2, :g2].mean((1, 2, 3)), t[:, :, :g2, g2:].mean((1, 2, 3)),
                               t[:, :, g2:, :g2].mean((1, 2, 3)), t[:, :, g2:, g2:].mean((1, 2, 3))], 1)
    return q(l1), q(cs)


@torch.inference_mode()
def batch_features_extra(cfg, model, rand_pred, tps, norm, Fr, E):
    """Horizon and quadrant features for one batch of TRs (frame ends E).

    Context = frames [e-CTX, e) (2 s, causal). The predictor forecasts n_hz = FUT2/2 future slots
    (0.5 s each, FUT2 = extra.fut_frames frames, default 16 = 4 s). Target/oracle = encoder over
    frames [e-CTX, e+FUT2), future-slot tokens. Returns numpy:
      hz_fore, hz_rand, hz_orac [B, n_hz, D]  per-slot token means
      hz_{fore,rand,copy}_{l1,cos} [B, n_hz]   per-slot errors vs layer-normed target (anchored at k)
      q_cmid, q_clast [B, 4, D]                quadrant means of the context (mid / last layer)
      q_fore, q_rand, q_orac [B, 4, D]         quadrant means over the first 2 s of future slots
      q_{fore,rand,copy}_{l1,cos} [B, 4]       quadrant errors over the first 2 s (anchored at k)
    """
    import numpy as np
    ex = cfg["extract"]
    CTX, FUT2 = ex["ctx_frames"], cfg.get("extra", {}).get("fut_frames", 16)
    cs, n_hz = CTX // 2, FUT2 // 2
    q_slots = ex["fut_frames"] // 2                      # 2 s = 4 slots for the quadrant features
    grid = int(round(tps ** 0.5))
    assert grid * grid == tps and grid % 2 == 0
    B, dev = len(E), norm.device
    for e in E:
        assert e - CTX >= 0 and e + FUT2 <= len(Fr)
    full = norm(np.stack([Fr[e - CTX:e + FUT2] for e in E]))
    past = full[:, :CTX].contiguous()                    # only frames before t_k
    h, mid = encode(model, past, cfg["model"]["layer_mid"])
    T_raw, _ = encode(model, full)
    T_raw = T_raw[:, cs * tps:(cs + n_hz) * tps]
    Tn = target_ln(T_raw)
    cm, tm = masks(B, cs, range(cs, cs + n_hz), tps, dev)
    P = predict(model.predictor, h, cm, tm)
    Pr = predict(rand_pred, h, cm, tm)
    assert P.shape == (B, n_hz * tps, h.shape[-1]), P.shape
    Cp = target_ln(h[:, (cs - 1) * tps:cs * tps]).repeat(1, n_hz, 1)
    D = h.shape[-1]
    f = lambda t: t.float().cpu().numpy()
    slot_mean = lambda x: x.float().reshape(B, n_hz, tps, D).mean(2)
    slot_l1 = lambda a, b: (a.float() - b.float()).abs().mean(-1).reshape(B, n_hz, tps).mean(2)
    slot_cos = lambda a, b: F.cosine_similarity(a.float(), b.float(), dim=-1).reshape(B, n_hz, tps).mean(2)
    out = {"hz_fore": f(slot_mean(P)), "hz_rand": f(slot_mean(Pr)), "hz_orac": f(slot_mean(T_raw))}
    for name, A in [("fore", P), ("rand", Pr), ("copy", Cp)]:
        out[f"hz_{name}_l1"], out[f"hz_{name}_cos"] = f(slot_l1(A, Tn)), f(slot_cos(A, Tn))
    nq = q_slots * tps
    out.update({"q_cmid": f(quadrant_pool(mid, cs, grid)), "q_clast": f(quadrant_pool(h, cs, grid)),
                "q_fore": f(quadrant_pool(P[:, :nq], q_slots, grid)), "q_rand": f(quadrant_pool(Pr[:, :nq], q_slots, grid)),
                "q_orac": f(quadrant_pool(T_raw[:, :nq], q_slots, grid))})
    for name, A in [("fore", P), ("rand", Pr), ("copy", Cp)]:
        l1, cs_ = quadrant_err(A[:, :nq], Tn[:, :nq], q_slots, grid)
        out[f"q_{name}_l1"], out[f"q_{name}_cos"] = f(l1), f(cs_)
    return out
