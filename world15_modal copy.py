"""
HunyuanWorld 1.5 — WorldPlay on Modal
Source: https://github.com/Tencent-Hunyuan/HY-WorldPlay
HF:     https://huggingface.co/tencent/HY-WorldPlay

INPUT:  Single image + text prompt + camera trajectory (pose string)
OUTPUT: MP4 video — AI-generated world from your camera path

IMPORTANT — what this is:
  WorldPlay generates a VIDEO of navigating a world from a starting image.
  The camera path is pre-planned via pose strings (e.g. "w-31" = move forward).
  This is NOT live keyboard control — that is Tencent's hosted product only.
  The distilled model (4 steps) is used — fastest, 28GB VRAM at sp=8.

Deploy:
    pip install modal && modal setup
    modal serve world15_modal.py
    modal deploy world15_modal.py

Requires HuggingFace token for FLUX.1-Redux-dev (gated model).
Create Modal secret:  modal secret create huggingface HF_TOKEN=<your_token>
Request FLUX access:  https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev

All parameters sourced from https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/main/README.md
"""

import modal

app = modal.App("hunyuan-world15")

hf_vol     = modal.Volume.from_name("world15-hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("world15-outputs",  create_if_missing=True)

HF_CACHE = "/hf_cache"
OUT_DIR  = "/outputs"
MODEL_BASE = "/models"  # HunyuanVideo-1.5 base + WorldPlay checkpoints

model_vol = modal.Volume.from_name("world15-models", create_if_missing=True)

# ── Image ─────────────────────────────────────────────────────────────────────
# From README: Python 3.10, requirements.txt, sageattention (required for WAN,
# optional for HunyuanVideo), flash-attn optional.
# We use the HunyuanVideo-based pipeline (recommended in README).
# ─────────────────────────────────────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "libgl1", "libglib2.0-0", "libgomp1",
        "ffmpeg",   # required by moviepy / video output
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Clone repo first so we can install from its requirements.txt
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay /app/HY-WorldPlay",
        "pip install -r /app/HY-WorldPlay/requirements.txt",
    )
    .pip_install(
        "sageattention",       # Required for WAN pipeline per README
        "huggingface_hub",
        "fastapi[standard]",
        "gradio==5.9.1",       # compatible version
    )
    # flash-attn optional — speeds up HunyuanVideo inference
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .env({
        "PYTHONPATH": "/app/HY-WorldPlay",
        "HF_HOME":    HF_CACHE,
    })
)


# ── Download function (runs once, cached in volume) ───────────────────────────
def download_worldplay_models(hf_token: str):
    """
    Downloads:
    1. HunyuanVideo-1.5 480P-I2V base model (from README download instructions)
    2. HY-WorldPlay checkpoints (ar_distilled_action_model — fastest, 4 steps)

    Per README: MODEL_PATH = HunyuanVideo-1.5 dir
    AR_DISTILL_ACTION_MODEL_PATH = ar_distilled_action_model/diffusion_pytorch_model.safetensors
    """
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_BASE, exist_ok=True)

    print("Downloading HunyuanVideo-1.5 base (480P-I2V)...")
    snapshot_download(
        repo_id="tencent/HunyuanVideo-1.5",
        allow_patterns=["transformer/480p_i2v/**", "vae/**", "scheduler/**"],
        local_dir=f"{MODEL_BASE}/HunyuanVideo-1.5",
        token=hf_token,
    )

    print("Downloading text encoders...")
    snapshot_download(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        local_dir=f"{MODEL_BASE}/HunyuanVideo-1.5/text_encoder/llm",
        token=hf_token,
    )
    snapshot_download(
        repo_id="google/byt5-small",
        local_dir=f"{MODEL_BASE}/HunyuanVideo-1.5/text_encoder/byt5-small",
        token=hf_token,
    )

    print("Downloading FLUX.1-Redux-dev vision encoder (requires HF access)...")
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-Redux-dev",
        allow_patterns=["*.safetensors", "*.json"],
        local_dir=f"{MODEL_BASE}/HunyuanVideo-1.5/vision_encoder/siglip",
        token=hf_token,
    )

    print("Downloading HY-WorldPlay distilled AR checkpoint...")
    snapshot_download(
        repo_id="tencent/HY-WorldPlay",
        allow_patterns=["ar_distilled_action_model/**"],
        local_dir=f"{MODEL_BASE}/HY-WorldPlay",
        token=hf_token,
    )
    print("All models downloaded.")


# ── Gradio App ────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    # Per README: sp=8 needs 28GB — A100 40GB fits with sp=8
    # sp=4 needs 34GB — also fits
    # We run sp=1 here (single GPU) = 72GB — needs A100 80GB
    # For 28GB mode, use torchrun with nproc=8 (not available in single-fn Modal)
    # Practical single-GPU: use distilled model + bf16 on A100 80GB
    gpu="A100:80GB",
    timeout=1200,
    volumes={
        HF_CACHE:   hf_vol,
        OUT_DIR:    output_vol,
        MODEL_BASE: model_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    min_containers=1,
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=3)
@modal.asgi_app()
def ui():
    import sys, os, subprocess, tempfile, shutil
    sys.path.insert(0, "/app/HY-WorldPlay")

    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    hf_token = os.environ.get("HF_TOKEN", "")

    # Download models once
    marker = f"{MODEL_BASE}/.downloaded"
    if not os.path.exists(marker):
        download_worldplay_models(hf_token)
        open(marker, "w").close()
        model_vol.commit()

    MODEL_PATH         = f"{MODEL_BASE}/HunyuanVideo-1.5"
    DISTILL_CKPT_PATH  = f"{MODEL_BASE}/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors"

    def run(image_file, prompt, pose, num_frames, seed, aspect_ratio):
        """
        Runs hyvideo/generate.py via subprocess exactly as the repo's run.sh does.

        Parameters per README/run.sh:
          --resolution     480p  (only available resolution)
          --few_step       true  (distilled model)
          --num_inference_steps 4  (distilled model uses 4 steps)
          --model_type     'ar'
          --pose           pose string e.g. "w-31"
          --video_length   num_frames (must satisfy (num_frames-1) % 4 == 0)
        """
        if image_file is None:
            return None, "❌ Please upload a starting image."

        # Validate num_frames constraint from README:
        # (num_frames-1) % 4 == 0  AND  [(num_frames-1)//4+1] % 4 == 0
        if (num_frames - 1) % 4 != 0:
            return None, f"❌ num_frames must satisfy (num_frames-1) % 4 == 0. Try 125, 97, 69, 41."
        ar_check = ((num_frames - 1) // 4 + 1) % 4
        if ar_check != 0:
            return None, f"❌ For AR model: [(num_frames-1)//4+1] must be divisible by 4. Use 125."

        out_path = tempfile.mkdtemp(dir=OUT_DIR, prefix="wp_")

        cmd = [
            "python", "/app/HY-WorldPlay/hyvideo/generate.py",
            "--prompt", prompt,
            "--image_path", image_file.name,
            "--resolution", "480p",
            "--aspect_ratio", aspect_ratio,
            "--video_length", str(num_frames),
            "--seed", str(seed),
            "--rewrite", "false",
            "--sr", "false",
            "--pose", pose,
            "--output_path", out_path,
            "--model_path", MODEL_PATH,
            "--action_ckpt", DISTILL_CKPT_PATH,
            "--few_step", "true",
            "--num_inference_steps", "4",
            "--model_type", "ar",
            "--use_vae_parallel", "false",
            "--use_sageattn", "false",
            "--use_fp8_gemm", "false",
            "--transformer_resident_ar_rollout", "true",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None, f"❌ Error:\n{result.stderr[-2000:]}"

        # Find output video
        for root, _, files in os.walk(out_path):
            for f in files:
                if f.endswith(".mp4"):
                    output_vol.commit()
                    return os.path.join(root, f), "✅ Done! Your world video is ready."

        return None, "❌ No video output found. Check logs."

    # ── UI ────────────────────────────────────────────────────────────────────
    with gr.Blocks(title="HunyuanWorld 1.5 — WorldPlay") as demo:
        gr.Markdown(
            "# 🎮 HunyuanWorld 1.5 — WorldPlay\n"
            "**Single image → AI-generated world video with camera trajectory control**\n\n"
            "Give it a starting image and a pose string, get back an MP4 of flying through an AI world."
        )
        with gr.Row():
            with gr.Column():
                img_in  = gr.File(label="Starting Image (required)", file_types=["image"])
                prompt  = gr.Textbox(
                    label="Scene Description (prompt)",
                    value="A serene park with stone bridge over calm water, lush trees",
                    lines=3,
                )
                gr.Markdown(
                    "### Camera Trajectory (Pose String)\n"
                    "Format: `action-duration` — controls how the camera moves.\n\n"
                    "| Code | Movement |\n|---|---|\n"
                    "| `w-N` | Forward N latents |\n"
                    "| `s-N` | Backward |\n"
                    "| `a-N` | Strafe left |\n"
                    "| `d-N` | Strafe right |\n"
                    "| `left-N` | Yaw left |\n"
                    "| `right-N` | Yaw right |\n"
                    "| `up-N` | Pitch up |\n"
                    "| `down-N` | Pitch down |\n\n"
                    "Chain them: `w-5, right-2, w-5` = forward, turn right, forward"
                )
                pose = gr.Textbox(
                    label="Pose String",
                    value="w-31",
                    info="Default: move forward 31 latents. Generates [1+31] latents = 125 frames.",
                )
                with gr.Row():
                    num_frames   = gr.Number(value=125, label="Num Frames (must satisfy (n-1)%4==0)",
                                             precision=0)
                    seed         = gr.Number(value=1, label="Seed", precision=0)
                    aspect_ratio = gr.Dropdown(["16:9", "4:3", "1:1"], value="16:9",
                                               label="Aspect Ratio")
                run_btn = gr.Button("🎬 Generate World Video", variant="primary", size="lg")
            with gr.Column():
                status   = gr.Textbox(label="Status", interactive=False, lines=4)
                video_out = gr.Video(label="Generated World (MP4)")

        run_btn.click(
            run,
            inputs=[img_in, prompt, pose, num_frames, seed, aspect_ratio],
            outputs=[video_out, status],
        )
        gr.Markdown(
            "---\n"
            "**Model:** distilled AR (4 steps) · **Resolution:** 480p only · "
            "**GPU:** A100 80GB · **Repo:** https://github.com/Tencent-Hunyuan/HY-WorldPlay\n\n"
            "⚠️ First run downloads ~30GB of models. Subsequent runs start in seconds."
        )

    demo.queue(max_size=3)
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
