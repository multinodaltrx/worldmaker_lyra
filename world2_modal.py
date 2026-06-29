"""
HY-World 2.0 — WorldMirror 2.0 on Modal
Source: https://github.com/Tencent-Hunyuan/HY-World-2.0
Docs:   https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/DOCUMENTATION.md

INPUT:  Multi-view images OR a video file (mp4/mov/avi)
OUTPUT: gaussians.ply, points.ply, depth maps, normal maps, camera_params.json

Deploy:
    pip install modal && modal setup
    modal serve world2_modal.py      # temporary URL for testing
    modal deploy world2_modal.py     # permanent URL

All parameters sourced exclusively from DOCUMENTATION.md — nothing invented.
"""

import modal

app = modal.App("hunyuan-world2")

# Persistent volumes — weights cached across runs, outputs stored permanently
hf_vol     = modal.Volume.from_name("world2-hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("world2-outputs",  create_if_missing=True)

HF_CACHE = "/hf_cache"
OUT_DIR  = "/outputs"

# ── Image ────────────────────────────────────────────────────────────────────
# Exact versions from requirements.txt:
#   torch==2.4.0, torchvision==0.19.0, CUDA 12.4, Python 3.10
#   gsplat wheel: gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl
# ─────────────────────────────────────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1", "wget")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        # Exact pins from requirements.txt
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
    # gsplat — exact URL from requirements.txt
    .pip_install(
        "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl"
    )
    # wheel required for flash-attn build with --no-build-isolation
    .pip_install("wheel", "setuptools", "packaging")
    # FlashAttention-2 — optional, speeds up ViT backbone
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    # Clone repo and add to path
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0 /app/HY-World-2.0",
    )
    .env({
        "PYTHONPATH": "/app/HY-World-2.0",
        "HF_HOME":    HF_CACHE,
    })
)


# ── Gradio App ────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A100",           # A100 40GB — single GPU, enable_bf16=True saves ~30% VRAM
    timeout=900,
    volumes={HF_CACHE: hf_vol, OUT_DIR: output_vol},
    min_containers=1,
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=5)
@modal.asgi_app()
def ui():
    import sys, os, zipfile, tempfile, shutil
    sys.path.insert(0, "/app/HY-World-2.0")

    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from huggingface_hub import snapshot_download

    # Download weights once into persistent volume
    weight_marker = os.path.join(HF_CACHE, "world2_downloaded.flag")
    if not os.path.exists(weight_marker):
        print("Downloading WorldMirror-2.0 weights...")
        snapshot_download(
            repo_id="tencent/HY-World-2.0",
            allow_patterns=["HY-WorldMirror-2.0/**"],
        )
        open(weight_marker, "w").close()
        hf_vol.commit()

    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    print("Loading WorldMirror-2.0 pipeline...")
    pipeline = WorldMirrorPipeline.from_pretrained(
        pretrained_model_name_or_path="tencent/HY-World-2.0",
        subfolder="HY-WorldMirror-2.0",
        use_fsdp=False,
        enable_bf16=True,
    )
    print("Pipeline ready.")

    def run(files, video, target_size, fps, video_max_frames,
            save_gs, save_depth, save_normal, save_camera,
            save_points, save_colmap, save_rendered):

        tmp = tempfile.mkdtemp(prefix="w2_input_")
        try:
            if video is not None:
                # Video path — pass directly to pipeline
                input_path = video
            elif files:
                for f in files:
                    shutil.copy(f.name, os.path.join(tmp, os.path.basename(f.name)))
                input_path = tmp
            else:
                return None, "❌ Upload images OR a video."

            result_dir = pipeline(
                input_path,
                output_path=OUT_DIR,
                # From DOCUMENTATION.md:
                target_size=int(target_size),   # max resolution, longest edge, multiple of 14
                fps=int(fps),                   # video frame extraction rate
                video_max_frames=int(video_max_frames),
                video_strategy="new",           # motion-aware extraction (repo default)
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
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        if not result_dir:
            return None, "❌ Pipeline returned no output."

        zip_path = tempfile.mktemp(suffix=".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, fnames in os.walk(result_dir):
                for fname in fnames:
                    fp = os.path.join(root, fname)
                    zf.write(fp, os.path.relpath(fp, result_dir))

        return zip_path, f"✅ Done — open gaussians.ply in SuperSplat: https://playcanvas.com/supersplat/editor"

    # ── UI layout ─────────────────────────────────────────────────────────────
    with gr.Blocks(title="HY-World 2.0 — WorldMirror") as demo:
        gr.Markdown(
            "# 🌍 HY-World 2.0 — WorldMirror 2.0\n"
            "**Multi-view images OR video → 3D Gaussian Splat + point cloud + depth + normals**\n\n"
            "Open `gaussians.ply` from the ZIP at **[SuperSplat](https://playcanvas.com/supersplat/editor)** to navigate your 3D world."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input — choose one")
                images_in = gr.File(
                    label="Option A: Multi-view images (8–20 overlapping photos)",
                    file_count="multiple",
                    file_types=["image"],
                )
                video_in = gr.Video(
                    label="Option B: Video (slow walkthrough — better results)",
                )
                with gr.Row():
                    target_size = gr.Slider(448, 1344, step=14, value=952,
                        label="Target Size (max edge px, multiple of 14)")
                    fps = gr.Slider(1, 10, step=1, value=2,
                        label="Video FPS (frames extracted/sec)")
                    max_frames = gr.Slider(4, 32, step=1, value=32,
                        label="Max Video Frames")
                with gr.Row():
                    save_gs      = gr.Checkbox(True,  label="gaussians.ply")
                    save_pts     = gr.Checkbox(True,  label="points.ply")
                    save_depth   = gr.Checkbox(True,  label="Depth maps")
                    save_normal  = gr.Checkbox(True,  label="Normal maps")
                    save_cam     = gr.Checkbox(True,  label="camera_params.json")
                    save_colmap  = gr.Checkbox(False, label="COLMAP format")
                    save_render  = gr.Checkbox(False, label="Rendered fly-through video")
                run_btn = gr.Button("🚀 Reconstruct 3D World", variant="primary", size="lg")
            with gr.Column():
                status = gr.Textbox(label="Status", interactive=False, lines=3)
                dl     = gr.File(label="⬇ Download outputs (ZIP)")

        run_btn.click(
            run,
            inputs=[images_in, video_in, target_size, fps, max_frames,
                    save_gs, save_depth, save_normal, save_cam,
                    save_pts, save_colmap, save_render],
            outputs=[dl, status],
        )
        gr.Markdown(
            "---\n**Model:** `tencent/HY-World-2.0` · WorldMirror-2.0 ~1.2B params · "
            "**GPU:** A100 40GB · **Repo:** https://github.com/Tencent-Hunyuan/HY-World-2.0"
        )

    demo.queue(max_size=5)
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
