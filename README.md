# Gemma-Assistantness

Ready-to-use **Assistant Axis** vectors for three generations of Gemma, so you don't have to run the expensive generate → activations → judge pipeline yourself.

| model | layers × d_model | suggested layer | source |
|---|---|---|---|
| `google/gemma-2-27b-it` | 46 × 4608 | 22 | copy of the paper's release ([MIT](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors)) |
| `google/gemma-3-27b-it` | 62 × 5376 | 31 | new: computed with the paper's 275-role protocol |
| `google/gemma-4-31B-it` | 60 × 5376 | 30 | new: computed with the paper's 275-role protocol |

The Assistant Axis comes from Lu et al. 2026, [arXiv:2601.10387](https://arxiv.org/abs/2601.10387). It is the direction in the residual stream separating the model's default Assistant persona from the 275 characters it can be prompted to play.

**Download (hosted on Hugging Face: [`timf34/gemma-assistant-axis-vectors`](https://huggingface.co/datasets/timf34/gemma-assistant-axis-vectors)):**

```bash
huggingface-cli download timf34/gemma-assistant-axis-vectors --repo-type dataset --local-dir .
```

## Files

```
vectors/<model>/assistant_axis.pt    [n_layers, d_model] bf16   default − mean(role vectors), every layer
vectors/<model>/default_vector.pt    [n_layers, d_model] bf16   mean activation under neutral system prompts
vectors/<model>/role_vectors/*.pt    275 × [n_layers, d_model]  per-role mean over fully in-role responses
```

```python
import torch
axis = torch.load("vectors/gemma-3-27b/assistant_axis.pt").float()   # [62, 5376]
u = axis[31] / axis[31].norm()
```

## One gotcha: z-score before projecting (Gemma 2 and 3)

At the middle layers, Gemma 2 and 3 have a few coordinates with massive activations, thousands of units against the usual tens. They dominate a raw dot product, so ranking roles by raw projection onto the axis gives nonsense. **Z-score per dimension using the role vectors' mean and std first.** In that space the axis is simply the default vector's z-position:

```python
import glob
R = torch.stack([torch.load(f).float()[31] for f in sorted(glob.glob("vectors/gemma-3-27b/role_vectors/*.pt"))])
mu, sd = R.mean(0), R.std(0) + 1e-8
d = (torch.load("vectors/gemma-3-27b/default_vector.pt").float()[31] - mu) / sd
assistantness = ((R - mu) / sd) @ (d / d.norm())
```

After this step, `assistant` ranks #1 of 275 in all three generations. The default Assistant sits +2.6, +3.0 and +2.6 SD from the role cloud for Gemma 2, 3 and 4.

## Caveats

- The Gemma 3/4 axes are validated geometrically: PC1 aligns with the axis, and the persona layout is consistent across generations. **They have not been validated with steering or activation capping yet.**
- Gemma 2's vectors average 1200 responses per role; Gemma 3/4 average 600.
- Gemma 4's axis matches `default − mean(roles)` to cos 0.998 rather than 1.000 (bf16 rounding).

Pipeline, raw responses, judge scores and the cross-generation analysis: [timf34/GemmaAssistantAxis](https://github.com/timf34/GemmaAssistantAxis) · [HF dataset](https://huggingface.co/datasets/timf34/gemma-assistant-axis-results).
