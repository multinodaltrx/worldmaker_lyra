"""
HunyuanVideo 1.5 — Text-to-Video on Modal
==========================================
Uses the HY-WorldPlay repo backbone (tencent/HunyuanVideo-1.5) for
text-to-video generation via Gradio on an A100 80GB GPU.

Sources:
  - https://github.com/Tencent-Hunyuan/HY-WorldPlay
  - https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/main/requirements.txt

Usage:
  modal deploy hunyuan_modal.py

Outputs:
  .mp4 video files saved to the hunyuan-outputs volume.
  Download: modal volume get hunyuan-outputs /outputs ./local_outputs
"""

import modal

# ---------------------------------------------------------------------------
# 1. App
# ---------------------------------------------------------------------------
app = modal.App("hunyuan-video-15")

# ---------------------------------------------------------------------------
# 2. Volumes
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("hunyuan-hf-cache", create_if_missing=True)
model_vol    = modal.Volume.from_name("hunyuan-models",   create_if_missing=True)
output_vol   = modal.Volume.from_name("hunyuan-outputs",  create_if_missing=True)

HF_CACHE_PATH = "/root/.cache/huggingface"
MODEL_PATH    = "/models/HunyuanVideo-1.5"
OUTPUT_PATH   = "/outputs"

# ---------------------------------------------------------------------------
# 3. Image — torch 2.6.0 + all deps from requirements.txt
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
    )
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "tqdm==4.67.1",
        "peft==0.17.0",
        "openai==2.8.0",
        "einops==0.8.0",
        "loguru==0.7.3",
        "numpy==1.26.4",
        "pillow==11.3.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "omegaconf>=2.3.0",
        "diffusers==0.35.0",
        "safetensors==0.4.5",
        "qwen-vl-utils==0.0.8",
        "huggingface-hub==0.34.0",
        "transformers[accelerate,tiktoken]==4.56.0",
        "wheel",
        "setuptools",
        "ninja",
        "pandas",
        "scipy",
        "protobuf",
        "moviepy==1.0.3",
        "ftfy",
        "gradio==5.49.1",
        "fastapi[standard]",
        "accelerate",
        "modelscope",
        "cloudpickle",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay /app/HY-WorldPlay",
    )
    .env({
        "PYTHONPATH": "/app/HY-WorldPlay",
        "HF_HOME":    HF_CACHE_PATH,
    })
)

# ---------------------------------------------------------------------------
# 4. Weight download (runs inside container on first cold start)
# ---------------------------------------------------------------------------
def download_models():
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_PATH, exist_ok=True)

    print("Downloading HunyuanVideo-1.5 backbone (~30 GB)...")
    snapshot_download(
        repo_id="tencent/HunyuanVideo-1.5",
        local_dir=MODEL_PATH,
        ignore_patterns=["*.git*", "*.md"],
    )

    print("Downloading Qwen2.5-VL-7B text encoder (~15 GB)...")
    snapshot_download(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        local_dir=f"{MODEL_PATH}/text_encoder/llm",
        ignore_patterns=["*.git*"],
    )

    print("Downloading byt5-small text encoder...")
    snapshot_download(
        repo_id="google/byt5-small",
        local_dir=f"{MODEL_PATH}/text_encoder/byt5-small",
    )

    print("All models downloaded.")


# ---------------------------------------------------------------------------
# 5. Gradio web app
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=modal.gpu.A100(size="80GB"),   # single GPU needs ~72 GB VRAM
    timeout=600,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        MODEL_PATH:    model_vol,
        OUTPUT_PATH:   output_vol,
    },
    min_containers=0,       # A100-80GB is expensive — spin down when idle
    scaledown_window=300,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def ui():
    import sys, os, subprocess, tempfile
    sys.path.insert(0, "/app/HY-WorldPlay")

    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    # Download weights on first cold start
    if not os.path.exists(f"{MODEL_PATH}/transformer"):
        download_models()
        model_vol.commit()

    def generate_video(prompt, num_frames, aspect_ratio, seed, num_steps):
        if not prompt.strip():
            return None, "Please enter a prompt."

        output_dir = tempfile.mkdtemp(prefix="hunyuan_out_")

        cmd = [
            "torchrun", "--nproc_per_node=1",
            "/app/HY-WorldPlay/hyvideo/generate.py",
            "--prompt",              prompt,
            "--resolution",          "720",
            "--aspect_ratio",        aspect_ratio,
            "--video_length",        str(int(num_frames)),
            "--seed",                str(int(seed)),
            "--output_path",         output_dir,
            "--model_path",          MODEL_PATH,
            "--few_step",            "true",
            "--num_inference_steps", str(int(num_steps)),
            "--model_type",          "ar",
            "--use_vae_parallel",    "false",
            "--use_sageattn",        "false",
            "--use_fp8_gemm",        "false",
            "--transformer_resident_ar_rollout", "true",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/app/HY-WorldPlay",
                timeout=540,
            )
        except subprocess.TimeoutExpired:
            return None, "Timed out after 9 minutes."

        if result.returncode != 0:
            return None, f"Error:\n{result.stderr[-3000:]}"

        # Find output video
        for fname in os.listdir(output_dir):
            if fname.endswith(".mp4"):
                out_path = os.path.join(output_dir, fname)
                output_vol.commit()
                return out_path, "Done!"

        return None, f"No output video found.\nSTDOUT: {result.stdout[-1000:]}"

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    with gr.Blocks(title="HunyuanVideo 1.5") as demo:
        gr.Markdown(
            "# HunyuanVideo 1.5 — Text-to-Video\n"
            "**GPU:** A100 80GB  |  **Model:** `tencent/HunyuanVideo-1.5`\n\n"
            "> **First request** downloads ~50 GB of weights — allow 10-15 min cold start."
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt_box = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="A gothic mansion at night, fog rolling through iron gates, cinematic lighting...",
                )
                with gr.Row():
                    num_frames = gr.Slider(
                        label="Frames  — must satisfy (N-1) % 4 == 0",
                        minimum=5, maximum=125, step=4, value=45,
                    )
                    aspect_ratio = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=["16:9", "9:16", "1:1"],
                        value="16:9",
                    )
                with gr.Row():
                    seed      = gr.Number(label="Seed", value=42)
                    num_steps = gr.Slider(
                        label="Steps  (4 = fast distilled, 50 = full quality)",
                        minimum=4, maximum=50, step=1, value=4,
                    )
                run_btn = gr.Button("Generate Video", variant="primary")

            with gr.Column(scale=1):
                status_box = gr.Textbox(label="Status", interactive=False)
                video_out  = gr.Video(label="Output Video")

        run_btn.click(
            fn=generate_video,
            inputs=[prompt_box, num_frames, aspect_ratio, seed, num_steps],
            outputs=[video_out, status_box],
        )

        gr.Markdown(
            "---\n"
            "**Repo:** https://github.com/Tencent-Hunyuan/HY-WorldPlay  \n"
            "**Download outputs:** `modal volume get hunyuan-outputs /outputs ./local_outputs`"
        )

    demo.queue(max_size=2)
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
