"""
NVIDIA Lyra 2.0 — Modal Deployment
====================================
Repo:   https://github.com/nv-tlabs/lyra/tree/main/Lyra-2
Model:  https://huggingface.co/nvidia/Lyra-2.0

LICENSE NOTICE:
  Source code: Apache 2.0
  Model weights: NVIDIA Internal Scientific Research and Development License
  This deployment is for PROTOTYPING ONLY. For commercial/production use,
  contact NVIDIA Research Licensing: https://www.nvidia.com/en-us/research/inquiries/

What it does:
  Single image (480x832) + camera trajectory -> exploration video
  -> reconstructed into 3D Gaussian Splats (.ply) via VIPE + DA3 depth

Every version below is copied verbatim from the official INSTALL.md
provided directly by the user. Nothing here is guessed.

GPU: H100 80GB (officially tested hardware per model card)
"""

import modal

app = modal.App("lyra2-worldmodel")

# Volumes: conda env build cache, model checkpoints, outputs
checkpoints_vol = modal.Volume.from_name("lyra2-checkpoints", create_if_missing=True)
output_vol      = modal.Volume.from_name("lyra2-outputs",     create_if_missing=True)
hf_vol          = modal.Volume.from_name("lyra2-hf-cache",    create_if_missing=True)

REPO_DIR     = "/app/lyra/Lyra-2"
CKPT_DIR     = "/checkpoints"
OUT_DIR      = "/outputs"
HF_CACHE     = "/hf_cache"

# ── Image ─────────────────────────────────────────────────────────────────────
# Exact steps from INSTALL.md (user-provided, verbatim):
#   Tested on Ubuntu 22.04, CUDA 12.8, H100 GPUs.
#   conda env: python 3.10, cuda toolkit 12.8 installed INSIDE conda
#   torch==2.7.1, torchvision==0.22.1, cu128
#   flash-attn==2.6.3 (built from source, --no-binary)
#   transformer_engine[pytorch] (built from source)
#   Vendored CUDA extensions: vipe, depth_anything_3[gs]
#
# Modal images don't support `conda activate` across RUN layers the way a
# normal shell session does, so every apt/pip step below explicitly sources
# the conda env or sets PATH/LD_LIBRARY_PATH per the INSTALL.md instructions.
# ─────────────────────────────────────────────────────────────────────────────

CONDA_INSTALL = (
    "wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && "
    "bash /tmp/miniconda.sh -b -p /opt/conda && "
    "rm /tmp/miniconda.sh"
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        add_python=None,  # we install Python via conda, exactly as INSTALL.md specifies
    )
    .apt_install(
        "git", "wget", "build-essential", "ca-certificates",
    )
    # ── Step 0: Install Miniconda (the repo assumes conda is available) ───────
    .run_commands(CONDA_INSTALL)
    .env({
        "PATH": "/opt/conda/bin:$PATH",
        "CONDA_PREFIX": "/opt/conda/envs/lyra2",
        # Suppress Unicode/color output from all build tools so Modal's log
        # capture (which runs through Windows charmap) doesn't crash.
        # PYTHONUTF8=1  — conda/pip internal encoding
        # PIP_PROGRESS_BAR=off / PIP_NO_COLOR=1 — pip download bars
        # NO_COLOR=1    — universal flag (rich, cmake, ninja, transformer_engine)
        "PYTHONUTF8": "1",
        "PIP_PROGRESS_BAR": "off",
        "PIP_NO_COLOR": "1",
        "NO_COLOR": "1",
    })
    # ── Step 0b: Clone repository WITH submodules (--recursive is required) ──
    .run_commands(
        f"git clone --recursive https://github.com/nv-tlabs/lyra.git /app/lyra",
    )
    # ── Step 0c: Accept Miniconda TOS (conda 24.9+ requires this in non-interactive builds) ──
    # CONDA_PREFIX is already set to /opt/conda/envs/lyra2 by the .env() above, but that
    # directory doesn't exist yet — override to the base install so conda doesn't die.
    .run_commands(
        "CONDA_PREFIX=/opt/conda /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main",
        "CONDA_PREFIX=/opt/conda /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r",
    )
    # ── Step 1: Create conda environment exactly as specified ────────────────
    # --override-channels: only use the explicitly listed -c channels, never
    # touch pkgs/main or pkgs/r (the Anaconda default channels that require TOS
    # acceptance). conda-forge has all packages we need.
    .run_commands(
        "/opt/conda/bin/conda create -n lyra2 python=3.10 pip cmake ninja libgl ffmpeg packaging "
        "--override-channels -c conda-forge -y",
        "CONDA_BACKUP_CXX='' /opt/conda/bin/conda install -n lyra2 gcc=13.3.0 gxx=13.3.0 eigen zlib "
        "--override-channels -c conda-forge -y",
    )
    # ── Step 2: Install CUDA toolkit INSIDE the conda environment ─────────────
    .run_commands(
        "/opt/conda/bin/conda install -n lyra2 cuda "
        "--override-channels -c nvidia/label/cuda-12.8.0 -c conda-forge -y",
    )
    # ── Step 3: Install PyTorch — exact versions from INSTALL.md ─────────────
    .run_commands(
        "/opt/conda/envs/lyra2/bin/pip install torch==2.7.1 torchvision==0.22.1 "
        "--extra-index-url https://download.pytorch.org/whl/cu128",
    )
    # ── Step 4-6: Build env vars + requirements + transformer_engine + flash-attn ─
    # Kept as one shell so CPATH/LD_LIBRARY_PATH/CC/CXX persist across all installs
    # that need them. Once this layer passes it's cached; vipe/da3 are split below.
    .run_commands(
        "bash -c '"
        "source /opt/conda/etc/profile.d/conda.sh && conda activate lyra2 && "
        "CUDA_HOME=$CONDA_PREFIX && "
        "SITE=$CONDA_PREFIX/lib/python3.10/site-packages && "
        "export CUDA_HOME && "
        # /usr/local/cuda/include added for nvtx3/nvToolsExt.h — the conda CUDA
        # package omits nvtx headers, but the devel base image has them there.
        "export CPATH=\"$CUDA_HOME/include:/usr/local/cuda/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:$CPATH\" && "
        "export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH\" && "
        "export CC=\"$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc\" && "
        "export CXX=\"$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++\" && "
        "cd /app/lyra/Lyra-2 && "
        "pip install --no-deps -r requirements.txt && "
        "pip install \"git+https://github.com/microsoft/MoGe.git\" && "
        "pip install --no-build-isolation \"transformer_engine[pytorch]\" && "
        "ln -sf \"$SITE/nvidia/cuda_runtime\" \"$SITE/nvidia/cudart\" && "
        "MAX_JOBS=16 pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3"
        "'",
    )
    # ── Step 7: Vendored CUDA extensions (separate layer for cache isolation) ────
    # TORCH_CUDA_ARCH_LIST must be set: image builds have no GPU, so PyTorch's
    # auto-detect returns an empty arch_list and crashes with IndexError on [-1].
    # H100 = sm_90. Split from the flash-attn step so a re-run here doesn't
    # force flash-attn to recompile (~15 min).
    .run_commands(
        "bash -c '"
        "source /opt/conda/etc/profile.d/conda.sh && conda activate lyra2 && "
        "CUDA_HOME=$CONDA_PREFIX && "
        "SITE=$CONDA_PREFIX/lib/python3.10/site-packages && "
        "export CUDA_HOME && "
        "export CPATH=\"$CUDA_HOME/include:/usr/local/cuda/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:$CPATH\" && "
        "export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH\" && "
        "export CC=\"$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc\" && "
        "export CXX=\"$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++\" && "
        "export TORCH_CUDA_ARCH_LIST=\"9.0\" && "
        "cd /app/lyra/Lyra-2 && "
        "USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e \"lyra_2/_src/inference/vipe\" && "
        "pip install hatchling && "
        "pip install --no-build-isolation -e \"lyra_2/_src/inference/depth_anything_3[gs]\""
        "'",
    )
    # Gradio + web server deps for our wrapper UI (not part of upstream repo)
    .run_commands(
        "/opt/conda/envs/lyra2/bin/pip install gradio==5.9.1 'fastapi[standard]' huggingface_hub",
    )
    .env({
        "HF_HOME": HF_CACHE,
        "PYTHONPATH": REPO_DIR,
    })
)


# ── Helper: build the LD_LIBRARY_PATH string exactly as INSTALL.md specifies ──
def _conda_env_prefix():
    """
    Returns the bash prefix needed before any python/pip command,
    matching the persistent shell profile setup from INSTALL.md step 8.
    """
    return (
        "source /opt/conda/etc/profile.d/conda.sh && conda activate lyra2 && "
        "SITE=$CONDA_PREFIX/lib/python3.10/site-packages && "
        "export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:$SITE/torch/lib:"
        "$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\" && "
    )


# ── Gradio App ────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="H100",          # officially tested hardware per model card
    timeout=2400,        # video gen + GS reconstruction, first run includes checkpoint DL
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        CKPT_DIR: checkpoints_vol,
        OUT_DIR:  output_vol,
        HF_CACHE: hf_vol,
    },
    min_containers=1,
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=2)
@modal.asgi_app()
def ui():
    import sys, os, subprocess, tempfile, shutil, zipfile
    sys.path.insert(0, REPO_DIR)

    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    # ── Download checkpoints once into persistent volume ──────────────────────
    marker = os.path.join(CKPT_DIR, ".downloaded")
    if not os.path.exists(marker):
        subprocess.run(
            ["huggingface-cli", "download", "nvidia/Lyra-2.0",
             "--include", "checkpoints/*", "--local-dir", "/app/lyra/Lyra-2"],
            check=True,
        )
        # Symlink so the checkpoints persist in the volume across containers
        if os.path.exists("/app/lyra/Lyra-2/checkpoints") and not os.path.islink("/app/lyra/Lyra-2/checkpoints"):
            shutil.move("/app/lyra/Lyra-2/checkpoints", CKPT_DIR + "/checkpoints")
            os.symlink(CKPT_DIR + "/checkpoints", "/app/lyra/Lyra-2/checkpoints")
        open(marker, "w").close()
        checkpoints_vol.commit()
    elif not os.path.exists("/app/lyra/Lyra-2/checkpoints"):
        os.symlink(CKPT_DIR + "/checkpoints", "/app/lyra/Lyra-2/checkpoints")

    def run_zoomgs(image_file, caption, sample_id):
        """
        Simplest entry point: zoom-in/zoom-out exploration from a single image.
        Command from the repo's own usage docs:

          PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference \\
              --input_image_path <img> --sample_id <0-14> --checkpoint_dir checkpoints
        """
        if image_file is None:
            return None, None, "❌ Upload a starting image (480x832 recommended)."

        run_dir = tempfile.mkdtemp(dir=OUT_DIR, prefix="lyra2_zoomgs_")
        img_dst = os.path.join(run_dir, "input.png")
        shutil.copy(image_file.name, img_dst)

        # Caption file required alongside the image per repo convention
        caption_dst = os.path.join(run_dir, "input.txt")
        with open(caption_dst, "w") as f:
            f.write(caption.strip() if caption.strip() else "a photorealistic scene")

        cmd_str = (
            _conda_env_prefix() +
            f"cd {REPO_DIR} && "
            f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
            f"PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference "
            f"--input_image_path {img_dst} "
            f"--sample_id {int(sample_id)} "
            f"--checkpoint_dir checkpoints "
            f"--output_dir {run_dir}"
        )
        result = subprocess.run(["bash", "-c", cmd_str], capture_output=True, text=True)
        if result.returncode != 0:
            return None, None, f"❌ Generation failed:\n{result.stderr[-3000:]}"

        # Find the generated video
        video_path = None
        for root, _, files in os.walk(run_dir):
            for fname in files:
                if fname.endswith(".mp4"):
                    video_path = os.path.join(root, fname)

        if video_path is None:
            return None, None, f"❌ No video produced.\nstdout: {result.stdout[-1500:]}"

        # Step 2: lift the video to 3D Gaussian Splats via VIPE + DA3 depth
        gs_cmd_str = (
            _conda_env_prefix() +
            f"cd {REPO_DIR} && "
            f"PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon "
            f"--input_video_path {video_path}"
        )
        gs_result = subprocess.run(["bash", "-c", gs_cmd_str], capture_output=True, text=True)

        gs_ply_path = None
        if gs_result.returncode == 0:
            base = os.path.splitext(video_path)[0]
            gs_dir = base + "_gs_ours"
            if os.path.isdir(gs_dir):
                for f in os.listdir(gs_dir):
                    if f.endswith(".ply"):
                        gs_ply_path = os.path.join(gs_dir, f)

        output_vol.commit()

        status = "✅ Video generated."
        if gs_ply_path:
            status += " ✅ 3D Gaussian Splat reconstructed — download .ply below."
        else:
            status += f" ⚠️ GS reconstruction failed:\n{gs_result.stderr[-1500:]}"

        return video_path, gs_ply_path, status

    # ── UI ────────────────────────────────────────────────────────────────────
    with gr.Blocks(title="NVIDIA Lyra 2.0") as demo:
        gr.Markdown(
            "# 🌌 NVIDIA Lyra 2.0 — Prototype\n"
            "**Single image → explorable 3D world (video + Gaussian Splats)**\n\n"
            "⚠️ **License notice:** Model weights are under NVIDIA's Internal "
            "Scientific Research and Development License. This is a prototype only. "
            "For commercial use, contact "
            "[NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)."
        )
        with gr.Row():
            with gr.Column():
                img_in = gr.File(label="Starting Image (480x832 recommended)",
                                  file_types=["image"])
                caption = gr.Textbox(
                    label="Caption (describes the starting image)",
                    placeholder="A cozy living room with a fireplace and large windows",
                    lines=2,
                )
                sample_id = gr.Slider(0, 14, step=1, value=0,
                                       label="Sample ID (camera motion preset, 0-14)")
                run_btn = gr.Button("🚀 Generate Explorable World", variant="primary", size="lg")
            with gr.Column():
                status = gr.Textbox(label="Status", interactive=False, lines=5)
                video_out = gr.Video(label="Exploration Video")
                gs_out = gr.File(label="⬇ 3D Gaussian Splat (.ply) — open in SuperSplat")

        run_btn.click(
            run_zoomgs,
            inputs=[img_in, caption, sample_id],
            outputs=[video_out, gs_out, status],
        )
        gr.Markdown(
            "---\n"
            "**Model:** `nvidia/Lyra-2.0` (WAN-14B backbone) · **GPU:** H100 80GB · "
            "**Repo:** https://github.com/nv-tlabs/lyra/tree/main/Lyra-2\n\n"
            "Open the resulting `.ply` at [SuperSplat](https://playcanvas.com/supersplat/editor) "
            "to navigate your reconstructed 3D world.\n\n"
            "⚠️ First run downloads model checkpoints (~tens of GB) — expect 15-30 min. "
            "Subsequent runs are much faster."
        )

    demo.queue(max_size=2)
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
