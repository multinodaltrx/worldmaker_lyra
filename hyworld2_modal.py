"""
HY-World 2.0 — WorldMirror 2.0 on Modal
========================================
Deploys the WorldMirror 2.0 pipeline (multi-view images / video -> 3D)
as a Gradio web app on Modal with an A100 40GB GPU.

Sources:
  - https://github.com/Tencent-Hunyuan/HY-World-2.0
  - https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/DOCUMENTATION.md
  - https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/requirements.txt

Usage:
  pip install modal && modal setup
  modal serve hyworld2_modal.py      # temporary URL
  modal deploy hyworld2_modal.py     # permanent URL

Outputs per inference run:
  gaussians.ply, points.ply, depth/*, normal/*, camera_params.json
"""

import modal

# ---------------------------------------------------------------------------
# 1. App
# ---------------------------------------------------------------------------
app = modal.App("hyworld2-worldmirror")

# ---------------------------------------------------------------------------
# 2. Volumes
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("hyworld2-hf-cache", create_if_missing=True)
output_vol   = modal.Volume.from_name("hyworld2-outputs",  create_if_missing=True)

HF_CACHE_PATH = "/root/.cache/huggingface"
OUTPUT_PATH   = "/outputs"

# ---------------------------------------------------------------------------
# 3. Image — exact versions from requirements.txt
#    Python 3.10, CUDA 12.4, torch 2.4.0 / torchvision 0.19.0
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
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
    # wheel/setuptools required by flash-attn's setup.py under --no-build-isolation
    .pip_install("wheel", "setuptools", "packaging")
    # FlashAttention-2 — needs torch installed first, no build isolation
    .pip_install("flash-attn", extra_options="--no-build-isolation")
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
# 4. Weight download helper (called once on cold start if volume is empty)
# ---------------------------------------------------------------------------
def download_weights():
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="tencent/HY-World-2.0",
        allow_patterns=["HY-WorldMirror-2.0/**"],
        local_dir=HF_CACHE_PATH + "/hub/tencent--HY-World-2.0",
    )
    print("WorldMirror-2.0 weights downloaded.")

# ---------------------------------------------------------------------------
# 5. WorldMirror cls — used by the CLI entrypoint
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={HF_CACHE_PATH: hf_cache_vol, OUTPUT_PATH: output_vol},
    min_containers=1,
    scaledown_window=300,
)
class WorldMirror:

    @modal.enter()
    def load_pipeline(self):
        import sys, os
        sys.path.insert(0, "/app/HY-World-2.0")
        weight_dir = os.path.join(HF_CACHE_PATH, "hub/tencent--HY-World-2.0/HY-WorldMirror-2.0")
        if not os.path.exists(weight_dir):
            download_weights()
            hf_cache_vol.commit()
        from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
        self.pipeline = WorldMirrorPipeline.from_pretrained(
            pretrained_model_name_or_path="tencent/HY-World-2.0",
            subfolder="HY-WorldMirror-2.0",
            use_fsdp=False,
            enable_bf16=True,
        )

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
        output_vol.commit()
        return result_path or OUTPUT_PATH

# ---------------------------------------------------------------------------
# 6. Gradio web app
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={HF_CACHE_PATH: hf_cache_vol, OUTPUT_PATH: output_vol},
    min_containers=1,
    scaledown_window=300,
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
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    weight_dir = os.path.join(HF_CACHE_PATH, "hub/tencent--HY-World-2.0/HY-WorldMirror-2.0")
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
        if not files:
            return None, "Please upload at least one image."

        tmp_input = tempfile.mkdtemp(prefix="wm2_input_")
        for f in files:
            # Gradio 5.x: FileData has .path; older has .name
            src = getattr(f, 'path', None) or getattr(f, 'name', None) or str(f)
            shutil.copy(src, os.path.join(tmp_input, os.path.basename(src)))

        try:
            result_dir = pipeline(
                tmp_input,
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
            return None, f"Error: {e}"

        zip_path = tempfile.mktemp(suffix=".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, fnames in os.walk(result_dir):
                for fname in fnames:
                    fpath = os.path.join(root, fname)
                    zf.write(fpath, os.path.relpath(fpath, result_dir))

        shutil.rmtree(tmp_input, ignore_errors=True)
        return zip_path, f"Done — outputs in {result_dir}"

    with gr.Blocks(title="HY-World 2.0 — WorldMirror 2.0") as demo:
        gr.Markdown(
            "# HY-World 2.0 — WorldMirror 2.0\n"
            "Upload **multi-view images** of a scene (8-20 overlapping photos). "
            "The model reconstructs a **3D Gaussian Splatting** scene, point cloud, "
            "depth maps, surface normals, and camera parameters.\n\n"
            "**Outputs** are zipped for download. "
            "Open `gaussians.ply` in [SuperSplat](https://playcanvas.com/supersplat/editor), "
            "Blender, Unity, or Unreal Engine."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.File(
                    label="Input Images",
                    file_count="multiple",
                    file_types=["image", ".mp4", ".mov", ".avi"],
                )
                target_size = gr.Slider(
                    label="Target Size (longest edge px, multiple of 14)",
                    minimum=448, maximum=1344, step=14, value=952,
                    info="Repo default: 952. Higher = more detail, more VRAM.",
                )
                with gr.Row():
                    save_gs       = gr.Checkbox(value=True,  label="gaussians.ply")
                    save_points   = gr.Checkbox(value=True,  label="points.ply")
                    save_depth    = gr.Checkbox(value=True,  label="Depth maps")
                    save_normal   = gr.Checkbox(value=True,  label="Normal maps")
                    save_camera   = gr.Checkbox(value=True,  label="camera_params.json")
                    save_colmap   = gr.Checkbox(value=False, label="COLMAP format")
                    save_rendered = gr.Checkbox(value=False, label="Rendered video")
                run_btn = gr.Button("Reconstruct 3D Scene", variant="primary", size="lg")

            with gr.Column(scale=1):
                status_box  = gr.Textbox(label="Status", interactive=False, lines=3)
                output_file = gr.File(label="Download Outputs (ZIP)")

        run_btn.click(
            fn=run_reconstruction,
            inputs=[image_input, target_size,
                    save_gs, save_depth, save_normal, save_camera,
                    save_points, save_colmap, save_rendered],
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
# 7. Headless CLI entrypoint
#    modal run hyworld2_modal.py --input ./my_images --target-size 952
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def reconstruct_cli(input: str = "./examples/worldrecon", target_size: int = 952):
    from pathlib import Path
    input_path = Path(input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    print(f"Uploading {input_path} to Modal...")
    model = WorldMirror()
    result = model.reconstruct.remote(input_path=str(input_path), target_size=target_size)
    print(f"Done. Output: {result}")
    print("Download: modal volume get hyworld2-outputs <path>")
