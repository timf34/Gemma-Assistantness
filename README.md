# Gemma-Assistantness

Precomputed [Assistant Axis](https://arxiv.org/abs/2601.10387) vectors for **Gemma 2, 3 and 4**, so you can experiment without running the pipeline yourself.

Vectors are on Hugging Face: [`timf34/gemma-assistant-axis-vectors`](https://huggingface.co/datasets/timf34/gemma-assistant-axis-vectors)

```bash
huggingface-cli download timf34/gemma-assistant-axis-vectors --repo-type dataset --local-dir .
```

| model | shape | middle layer |
|---|---|---|
| `google/gemma-2-27b-it` | 46 × 4608 | 22 |
| `google/gemma-3-27b-it` | 62 × 5376 | 31 |
| `google/gemma-4-31B-it` | 60 × 5376 | 30 |

Each model folder has `assistant_axis.pt`, `default_vector.pt` and `role_vectors/` (275 personas), all shaped `[n_layers, d_model]`.

```python
import torch
axis = torch.load("vectors/gemma-3-27b/assistant_axis.pt").float()
```

**Tip:** Gemma 2 and 3 have a few huge-activation dimensions, so z-score the role vectors per dimension before projecting onto the axis. Otherwise the rankings are nonsense.

Gemma 2 vectors are from the [original release](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors) (MIT). Gemma 3/4 were computed with the same protocol ([pipeline](https://github.com/timf34/GemmaAssistantAxis)).
