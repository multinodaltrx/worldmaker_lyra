"""
HY-World 2.0 — WorldMirror 2.0 on Modal
========================================
Deploys the WorldMirror 2.0 pipeline (multi-view images / video → 3D)
as a Gradio web app on Modal with an A100 40GB GPU.

Sources:
  - https://github.com/Tencent-Hunyuan/HY-World-2.0
  - https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/DOCUMENTATION.md
  - https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/requirements.txt

Usage:
  # Install Modal locally first:
  pip install modal
  modal setup               # authenticate

  # Dev mode (temporary URL, hot-reload):
  modal serve hyworld2_modal.py

  # Permanent deploy:
  modal deploy hyworld2_modal.py

Outputs (per inference run):
  - gaussians.ply           (3D Gaussian Splatting — open in Blender/Unity/UE)
  - points.ply              (coloured point cloud)
  - depth/depth_NNNN.png    (depth maps)
  - normal/normal_NNNN.png  (surface normal maps)
  - camera_params.json      (camera extrinsics + intrinsics)

HuggingFace token:
  The model is public (tencent/HY-World-2.0) so no token is required.
  If it ever becomes gated, create a Modal secret:
    modal secret create huggingface HF_TOKEN=<your_token>
  and add secrets=[modal.Secret.from_name("huggingface")] to the @app.cls decorator.
"""

import modal

# ---------------------------------------------------------------------------
# 1. App definition
# ---------------------------------------------------------------------------
app = modal.App("hyworld2-worldmirror")

# ---------------------------------------------------------------------------
# 2. Persistent volume — caches HuggingFace model weights (~5–10 GB) so they
#    are only downloaded once across all container starts.
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("hyworld2-hf-cache", create_if_missing=True)
output_vol   = modal.Volume.from_name("hyworld2-outputs",  create_if_missing=True)

HF_CACHE_PATH  = "/root/.cache/huggingface"
OUTPUT_PATH    = "/outputs"

# ---------------------------------------------------------------------------
# 3. Container image
#
# Exact versions from the repo's requirements.txt:
#   - Python 3.10  (gsplat wheel is built for cp310)
#   - CUDA 12.4    (gsplat wheel suffix: pt24cu124)
#   - torch 2.4.0
#   - torchvision 0.19.0
#   - All other deps from requirements.txt
#
# FlashAttention-2 is installed last (needs torch first, no build isolation).
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        # Official NVIDIA image: CUDA 12.4 + Ubuntu 22.04
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "libgl1",           # OpenCV runtime dependency
        "libglib2.0-0",     # OpenCV runtime dependency
        "libgomp1",         # open3d / OpenMP
    )
    # Install PyTorch first (exact versions from repo)
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Install all deps from requirements.txt (exact versions where pinned)
    .pip_install(
        "gradio==5.49.1",
        "moviepy==1.0.3",
        "jaxtyping",
        "typeguard",
        "tqdm",
        "omegaconf",
        "plyfile",
        "colorspacious",
        "pydantic",
        "opencv-python",
        "scipy",
        "requests",
        "trimesh",
        "matplotlib",
        "spaces",
        "pillow-heif",
        "onnxruntime",
        "einops",
        "torchmetrics",
        "uniception",
        "safetensors",
        "numpy<2.0.0",
        "open3d==0.18.0",
        "pycolmap==3.10.0",
        "huggingface_hub",
        "fastapi[standard]",
    )
    # gsplat pre-built wheel — exact URL from requirements.txt
    .pip_install(
        "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl"
    )
    # wheel/setuptools required for flash-attn build with --no-build-isolation
    .pip_install("wheel", "setuptools", "packaging")
    # FlashAttention-2 (optional but recommended — speeds up ViT backbone)
    .pip_install(
        "flash-attn",
        extra_options="--no-build-isolation",
    )
    # Clone the HY-World 2.0 repo into the image
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0 /app/HY-World-2.0",
        "pip install -e /app/HY-World-2.0 --no-deps 2>/dev/null || true",
    )
    .env({
        "PYTHONPATH": "/app/HY-World-2.0",
        "HF_HOME":    HF_CACHE_PATH,
    })
)

# ---------------------------------------------------------------------------
# 4. Model weight pre-download (runs once at image build / volume population)
#    Uses huggingface_hub.snapshot_download to pull the WorldMirror-2.0 weights
#    into the persistent volume, not baked into the image layer.
# ---------------------------------------------------------------------------
def download_weights():
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="tencent/HY-World-2.0",
        # Only pull the WorldMirror-2.0 subfolder — skips other large assets
        allow_patterns=["HY-WorldMirror-2.0/**"],
        local_dir=HF_CACHE_PATH + "/hub/tencent--HY-World-2.0",
    )
    print("✅  WorldMirror-2.0 weights downloaded.")

# ---------------------------------------------------------------------------
# 5. WorldMirror inference class
#    Decorated with @modal.cls so Modal keeps the pipeline warm between calls.
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="A100",          # A100 40GB — minimum recommended for single-GPU mode
    timeout=600,         # 10 min per inference call
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        OUTPUT_PATH:   output_vol,
    },
    # Keep one container warm to avoid cold-start on every request
    min_containers=1,
    scaledown_window=300,
)
class WorldMirror:

    @modal.enter()
    def load_pipeline(self):
        """Load once when the container starts (warm pool)."""
        import sys
        sys.path.insert(0, "/app/HY-World-2.0")

        # Download weights on first run if not yet cached
        import os
        weight_dir = os.path.join(
            HF_CACHE_PATH,
            "hub/tencent--HY-World-2.0/HY-WorldMirror-2.0",
        )
        if not os.path.exists(weight_dir):
            download_weights()
            hf_cache_vol.commit()

        from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
        self.pipeline = WorldMirrorPipeline.from_pretrained(
            pretrained_model_name_or_path="tencent/HY-World-2.0",
            subfolder="HY-WorldMirror-2.0",
            # Single GPU defaults — no FSDP needed for A100 40GB
            use_fsdp=False,
            enable_bf16=True,   # Cuts VRAM ~30%
        )
        print("✅  WorldMirror 2.0 pipeline loaded.")

    @modal.method()
    def reconstruct(
        self,
        input_path: str,
        target_size: int = 952,
        save_gs: bool = True,
        save_depth: bool = True,
        save_normal: bool = True,
        save_camera: bool = True,
        save_points: bool = True,
        save_colmap: bool = False,
        save_rendered: bool = False,
        compress_pts: bool = True,
    ) -> str:
        """
        Run WorldMirror reconstruction.

        Parameters match WorldMirrorPipeline.__call__ exactly as documented in
        DOCUMENTATION.md.

        Args:
            input_path:   Path (inside the container) to a directory of images
                          OR a video file. Must be reachable from the container.
            target_size:  Max inference resolution (longest edge).
                          Repo default: 952. Resized + center-cropped to nearest
                          multiple of 14.
            save_gs:      Save gaussians.ply (3DGS output).
            save_depth:   Save per-view depth maps (PNG + NPY).
            save_normal:  Save per-view surface normal maps (PNG).
            save_camera:  Save camera_params.json.
            save_points:  Save depth-derived point cloud (points.ply).
            save_colmap:  Save COLMAP sparse reconstruction.
            save_rendered: Render interpolated fly-through video.
            compress_pts: Voxel-merge + subsample the point cloud.

        Returns:
            Path to the output directory inside the container.
        """
        import os
        result_path = self.pipeline(
            input_path,
            output_path=OUTPUT_PATH,
            target_size=target_size,
            save_gs=save_gs,
            save_depth=save_depth,
            save_normal=save_normal,
            save_camera=save_camera,
            save_points=save_points,
            save_colmap=save_colmap,
            save_rendered=save_rendered,
            compress_pts=compress_pts,
        )
        # Commit output volume so files persist after container exits
        output_vol.commit()
        return result_path or OUTPUT_PATH


# ---------------------------------------------------------------------------
# 6. Gradio web app
#    Matches the upstream gradio_app interface described in DOCUMENTATION.md.
#    Users upload images (or a video), click Run, and download the .ply outputs.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    # Gradio needs a GPU for the WorldMirror calls it proxies
    gpu="A100",
    timeout=600,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        OUTPUT_PATH:   output_vol,
    },
    min_containers=1,
    scaledown_window=300,
    # Sticky sessions required by Gradio
    max_containers=1,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def ui():
    import sys, os, zipfile, tempfile, shutil
    sys.path.insert(0, "/app/HY-World-2.0")

    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    # Lazy-load the pipeline inside the web process
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    # Download weights if not cached
    weight_dir = os.path.join(
        HF_CACHE_PATH,
        "hub/tencent--HY-World-2.0/HY-WorldMirror-2.0",
    )
    if not os.path.exists(weight_dir):
        download_weights()
        hf_cache_vol.commit()

    pipeline = WorldMirrorPipeline.from_pretrained(
        pretrained_model_name_or_path="tencent/HY-World-2.0",
        subfolder="HY-WorldMirror-2.0",
        use_fsdp=False,
        enable_bf16=True,
    )

    def run_reconstruction(files, target_size, save_gs, save_depth,
                           save_normal, save_camera, save_points,
                           save_colmap, save_rendered):
        """
        Gradio callback.  Accepts uploaded image files, runs the pipeline,
        and returns a ZIP of all outputs for download.
        """
        if not files:
            return None, "❌ Please upload at least one image."

        # Copy uploaded files into a temp dir
        # Gradio 5.x returns FileData objects with .path; older versions use .name
        tmp_input = tempfile.mkdtemp(prefix="worldmirror_input_")
        for f in files:
            src = getattr(f, 'path', None) or getattr(f, 'name', None) or str(f)
            shutil.copy(src, os.path.join(tmp_input, os.path.basename(src)))

        # If a single video file was uploaded, pass it directly (not as a dir)
        copied = os.listdir(tmp_input)
        video_exts = {'.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI'}
        if len(copied) == 1 and os.path.splitext(copied[0])[1] in video_exts:
            input_path = os.path.join(tmp_input, copied[0])
        else:
            input_path = tmp_input

        try:
            result_dir = pipeline(
                input_path,
                output_path=OUTPUT_PATH,
                target_size=int(target_size),
                save_gs=save_gs,
                save_depth=save_depth,
                save_normal=save_normal,
                save_camera=save_camera,
                save_points=save_points,
                save_colmap=save_colmap,
                save_rendered=save_rendered,
                compress_pts=True,
            )
            output_vol.commit()
        except Exception as e:
            shutil.rmtree(tmp_input, ignore_errors=True)
            return None, f"❌ Error: {e}"

        # Pack output directory into a ZIP for download
        zip_path = tempfile.mktemp(suffix=".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, fnames in os.walk(result_dir):
                for fname in fnames:
                    fpath = os.path.join(root, fname)
                    zf.write(fpath, os.path.relpath(fpath, result_dir))

        shutil.rmtree(tmp_input, ignore_errors=True)
        return zip_path, f"✅ Done — outputs in {result_dir}"

    # ------------------------------------------------------------------
    # Gradio UI
    # ------------------------------------------------------------------
    with gr.Blocks(title="HY-World 2.0 — WorldMirror 2.0") as demo:
        gr.Markdown(
            "# HY-World 2.0 — WorldMirror 2.0\n"
            "Upload **multi-view images** (or a short video) of a scene. "
            "The model reconstructs a **3D Gaussian Splatting** scene, point cloud, "
            "depth maps, surface normals, and camera parameters in a single forward pass.\n\n"
            "**Outputs** are zipped for download. Open `gaussians.ply` in "
            "[SuperSplat](https://playcanvas.com/supersplat/editor), Blender, Unity, or Unreal Engine."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.File(
                    label="Input Images (or Video)",
                    file_count="multiple",
                    file_types=["image", ".mp4", ".mov", ".avi"],
                )
                target_size = gr.Slider(
                    label="Target Size (max resolution, longest edge)",
                    minimum=448,
                    maximum=1344,
                    step=14,
                    value=952,   # repo default
                    info="Images resized + center-cropped to nearest multiple of 14",
                )
                with gr.Row():
                    save_gs      = gr.Checkbox(label="gaussians.ply",      value=True)
                    save_points  = gr.Checkbox(label="points.ply",          value=True)
                    save_depth   = gr.Checkbox(label="Depth maps",          value=True)
                    save_normal  = gr.Checkbox(label="Normal maps",         value=True)
                    save_camera  = gr.Checkbox(label="camera_params.json",  value=True)
                    save_colmap  = gr.Checkbox(label="COLMAP format",       value=False)
                    save_rendered = gr.Checkbox(label="Rendered video",     value=False)

                run_btn = gr.Button("🚀 Reconstruct", variant="primary")

            with gr.Column(scale=1):
                status_box  = gr.Textbox(label="Status", interactive=False)
                output_file = gr.File(label="Download Outputs (ZIP)")

        run_btn.click(
            fn=run_reconstruction,
            inputs=[
                image_input, target_size,
                save_gs, save_depth, save_normal, save_camera,
                save_points, save_colmap, save_rendered,
            ],
            outputs=[output_file, status_box],
        )

        gr.Markdown(
            "---\n"
            "**Model:** `tencent/HY-World-2.0` — WorldMirror-2.0 (~1.2B params)  \n"
            "**GPU:** A100 40GB  \n"
            "**Repo:** https://github.com/Tencent-Hunyuan/HY-World-2.0"
        )

    demo.queue(max_size=5)
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")


# ---------------------------------------------------------------------------
# 7. Optional: headless CLI entrypoint
#    Run locally against Modal GPU:
#      modal run hyworld2_modal.py::reconstruct_cli --input /path/to/images
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def reconstruct_cli(input: str = "./examples/worldrecon", target_size: int = 952):
    """
    Headless CLI — runs WorldMirror on Modal from your local machine.

    Args:
        input:       Local path to image directory. Uploaded to Modal at runtime.
        target_size: Max inference resolution (default: 952, repo default).

    Example:
        modal run hyworld2_modal.py --input ./my_images --target-size 952
    """
    import os
    from pathlib import Path

    input_path = Path(input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"📤 Uploading {input_path} to Modal...")
    model = WorldMirror()
    result = model.reconstruct.remote(
        input_path=str(input_path),
        target_size=target_size,
    )
    print(f"✅ Reconstruction complete. Output directory: {result}")
    print("   Download outputs with: modal volume get hyworld2-outputs <path>")
