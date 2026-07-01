# NVIDIA Lyra 2.0 — Modal Deployment

**Single image → explorable 3D world (zoom video + Gaussian Splat `.ply`)**

> **License:** Model weights are under the *NVIDIA Internal Scientific Research and Development License*.  
> This deployment is for **prototyping / internal demo only**.  
> For commercial or production use, contact [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

---

## What it does

Upload one photo and Lyra 2.0 generates:

| Output | Format | What you get |
|--------|--------|--------------|
| Exploration video | `.mp4` | Smooth zoom-in / zoom-out fly-through |
| 3D Gaussian Splat | `.ply` | Navigable 3D scene — open in [SuperSplat](https://playcanvas.com/supersplat/editor) |

The model runs on an **H100 80 GB GPU** on Modal's serverless cloud.  
First run downloads ~30 GB of model checkpoints (takes 15–30 min).  
All subsequent runs start in seconds.

---

## Accessing the live deployment

If the app is already deployed, open the public URL in any browser:

```
https://multinodaltrx--lyra2-worldmodel-ui.modal.run
```

1. Upload an image (recommended resolution: **480 × 832 px**).
2. Write a short caption describing the scene.
3. Pick a *Sample ID* (0–14) — each preset gives a slightly different camera motion.
4. Click **Generate Explorable World** and wait (≈ 3–8 min on a warm container).
5. Download the video and / or the `.ply` Gaussian Splat file.

To view the `.ply` interactively, drag it into [SuperSplat](https://playcanvas.com/supersplat/editor) — no install needed, runs in the browser.

---

## Deploying from scratch (for developers)

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| [Modal account](https://modal.com/) | Free tier works; you need a Pro plan for H100 GPU access |
| [Modal CLI](https://modal.com/docs/guide) | `pip install modal` then `modal setup` |
| [HuggingFace account](https://huggingface.co/) | Accept the [Lyra-2.0 model licence](https://huggingface.co/nvidia/Lyra-2.0) |
| HuggingFace token | Must have *read* permission for the `nvidia/Lyra-2.0` repo |

### 1 — Clone

```bash
git clone https://github.com/multinodaltrx/worldmaker_lyra.git
cd worldmaker_lyra
```

### 2 — Create the Modal HuggingFace secret

```bash
modal secret create huggingface-secret HUGGING_FACE_HUB_TOKEN=hf_YOUR_TOKEN_HERE
```

Replace `hf_YOUR_TOKEN_HERE` with your HuggingFace read token (get one at  
https://huggingface.co/settings/tokens).

### 3 — Deploy

```bash
modal deploy lyra2_modal.py
```

Modal will:
- Build the Docker image (takes **45–90 min** on first build — CUDA extensions compile from source)
- Print a public URL when done

Subsequent deploys re-use cached image layers and finish in **< 30 seconds**.

### 4 — Open the app

```
https://<your-modal-workspace>--lyra2-worldmodel-ui.modal.run
```

---

## Volumes (persistent storage)

Three Modal volumes are created automatically and persist across runs:

| Volume name | Mount | Purpose |
|-------------|-------|---------|
| `lyra2-checkpoints` | `/checkpoints` | Downloaded model weights (~30 GB) |
| `lyra2-outputs` | `/outputs` | Generated videos and `.ply` files |
| `lyra2-hf-cache` | `/hf_cache` | HuggingFace model cache |

To inspect outputs from the CLI:

```bash
modal volume ls lyra2-outputs
modal volume get lyra2-outputs <path/to/file> ./local_copy
```

---

## Updating the deployment

Edit `lyra2_modal.py`, then:

```bash
modal deploy lyra2_modal.py
```

---

## Model card

| Field | Value |
|-------|-------|
| Model | `nvidia/Lyra-2.0` |
| Backbone | WAN-14B |
| Tested GPU | H100 80 GB |
| Framework | PyTorch 2.7.1 + CUDA 12.8 |
| Upstream repo | https://github.com/nv-tlabs/lyra/tree/main/Lyra-2 |
