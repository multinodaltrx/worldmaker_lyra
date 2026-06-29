"""
HunyuanWorld 1.0 on Modal
Source: https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0

INPUT:  Single image OR text prompt
OUTPUT: Full 3D world mesh (glb/obj) + panorama + modelviewer.html for browser play

Two-stage pipeline per README:
  Stage 1 — demo_panogen.py  → generates 360° panorama from image or text
  Stage 2 — demo_scenegen.py → panorama → layered 3D world mesh

Deploy:
    pip install modal && modal setup
    modal serve world10_modal.py
    modal deploy world10_modal.py

Requires HuggingFace login (models at tencent/HunyuanWorld-1):
    modal secret create huggingface HF_TOKEN=<your_token>

All parameters from https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0/blob/main/README.md
"""

import modal

app = modal.App("hunyuan-world10")

hf_vol     = modal.Volume.from_name("world10-hf-cache",  create_if_missing=True)
output_vol = modal.Volume.from_name("world10-outputs",    create_if_missing=True)
model_vol  = modal.Volume.from_name("world10-models",     create_if_missing=True)

HF_CACHE   = "/hf_cache"
OUT_DIR    = "/outputs"
REPO_DIR   = "/app/HunyuanWorld-1.0"
MODEL_DIR  = "/models"

# ── Image ─────────────────────────────────────────────────────────────────────
# From docker/HunyuanWorld.yaml: Python 3.10, PyTorch 2.5.0+cu124
# NOTE: the repo has NO requirements.txt — deps are in docker/HunyuanWorld.yaml (conda).
# We install the pip section of that yaml manually.
# ─────────────────────────────────────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "libgl1", "libglib2.0-0", "libgomp1",
        "cmake", "build-essential",   # for draco
        "wget", "unzip", "ffmpeg",
    )
    .pip_install(
        "torch==2.5.0",
        "torchvision==0.20.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Core packages from docker/HunyuanWorld.yaml pip section
    .pip_install(
        "transformers==4.51.0",
        "diffusers==0.34.0",
        "accelerate==1.6.0",
        "huggingface-hub==0.30.2",
        "safetensors==0.5.3",
        "peft==0.15.0",
        "einops==0.4.1",
        "omegaconf==2.1.2",
        "kornia==0.8.0",
        "timm==1.0.13",
        "pytorch-lightning==2.4.0",
        "torchmetrics==1.7.1",
        "opencv-python-headless",
        "pillow",
        "numpy==1.24.1",
        "scipy",
        "scikit-image",
        "imageio",
        "imageio-ffmpeg",
        "matplotlib",
        "plyfile==1.1",
        "open3d>=0.18.0",
        "trimesh>=4.6.1",
        "py360convert",
        "onnxruntime-gpu",
        "easydict",
        "addict",
        "loguru",
        "ftfy",
        "pandas",
        "h5py",
        "tqdm",
        "packaging",
    )
    # MoGe depth model — also installs utils3d as a dependency
    .run_commands(
        "pip install git+https://github.com/microsoft/MoGe.git",
    )
    # Clone the repo (no requirements.txt in repo — all deps installed above)
    .run_commands(
        f"git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0 {REPO_DIR}",
    )
    # Real-ESRGAN (required by README)
    .run_commands(
        "git clone https://github.com/xinntao/Real-ESRGAN.git /app/Real-ESRGAN",
        "cd /app/Real-ESRGAN && pip install basicsr-fixed facexlib gfpgan",
        "cd /app/Real-ESRGAN && pip install -r requirements.txt",
        "cd /app/Real-ESRGAN && python setup.py develop",
    )
    # Patch basicsr — functional_tensor was removed in torchvision>=0.17, path is fixed in cuda+py3.10 image
    .run_commands(
        "sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/"
        "from torchvision.transforms.functional import rgb_to_grayscale/g' "
        "/usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py",
    )
    # ZIM segmentation (required by README)
    .run_commands(
        "git clone https://github.com/naver-ai/ZIM.git /app/ZIM",
        "cd /app/ZIM && pip install -e .",
        "mkdir -p /app/ZIM/zim_vit_l_2092",
        "wget -q https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx -O /app/ZIM/zim_vit_l_2092/encoder.onnx",
        "wget -q https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx -O /app/ZIM/zim_vit_l_2092/decoder.onnx",
    )
    # Draco (3D mesh compression)
    .run_commands(
        "git clone https://github.com/google/draco.git /tmp/draco",
        "cd /tmp/draco && mkdir build && cd build && cmake .. && make -j4 && make install",
        "rm -rf /tmp/draco",
    )
    # wheel needed for flash-attn --no-build-isolation
    .pip_install("wheel", "setuptools")
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    # Force-reinstall packages Real-ESRGAN/ZIM may have clobbered
    .pip_install(
        "transformers==4.51.0",
        "diffusers==0.34.0",
        "accelerate==1.6.0",
        "gradio==5.49.1",
        "fastapi[standard]",
    )
    .env({
        "PYTHONPATH": REPO_DIR,
        "HF_HOME":    HF_CACHE,
        "ESRGAN_PATH": "/app/Real-ESRGAN",
        "ZIM_PATH":    "/app/ZIM",
    })
)


# ── Gradio App ────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    # HunyuanWorld-1.0 uses Flux-based model — A100 40GB is sufficient
    # --fp8_gemm --fp8_attention flags available per README for memory saving
    gpu="A100",
    timeout=1800,   # panorama + scene gen can take 20-30 min on first run
    volumes={
        HF_CACHE:  hf_vol,
        OUT_DIR:   output_vol,
        MODEL_DIR: model_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    min_containers=1,
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=3)
@modal.asgi_app()
def ui():
    import sys, os, subprocess, tempfile, shutil, zipfile
    sys.path.insert(0, REPO_DIR)

    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    hf_token = os.environ.get("HF_TOKEN", "")

    # Login to HuggingFace so the scripts can download model weights
    if hf_token:
        subprocess.run(
            ["huggingface-cli", "login", "--token", hf_token],
            capture_output=True,
        )

    def run_world(
        mode,           # "Image to World" or "Text to World"
        image_file,
        prompt,
        labels_fg1,     # space-separated foreground object labels (layer 1)
        labels_fg2,     # space-separated foreground object labels (layer 2)
        scene_class,    # "outdoor" or "indoor"
        use_fp8,        # use --fp8_gemm --fp8_attention for memory saving
    ):
        """
        Runs the two-stage pipeline per README:
          Stage 1: demo_panogen.py  → panorama
          Stage 2: demo_scenegen.py → 3D world

        Per README exact CLI:
          python3 demo_panogen.py --prompt "" --image_path <img> --output_path <out>
          python3 demo_scenegen.py --image_path <panorama.png> --labels_fg1 stones
                                   --labels_fg2 trees --classes outdoor --output_path <out>

        fp8 flags from README quantization section:
          --fp8_gemm --fp8_attention
        """
        out_path = tempfile.mkdtemp(dir=OUT_DIR, prefix="hw10_")

        # ── Stage 1: Panorama generation ─────────────────────────────────────
        pano_cmd = [
            "python3", f"{REPO_DIR}/demo_panogen.py",
            "--output_path", out_path,
        ]
        if mode == "Image to World":
            if image_file is None:
                return None, "❌ Please upload an image."
            img_path = getattr(image_file, 'path', None) or getattr(image_file, 'name', None) or str(image_file)
            pano_cmd += ["--prompt", "", "--image_path", img_path]
        else:
            if not prompt.strip():
                return None, "❌ Please enter a text prompt."
            pano_cmd += ["--prompt", prompt.strip()]

        if use_fp8:
            pano_cmd += ["--fp8_gemm", "--fp8_attention"]

        print(f"Stage 1 command: {' '.join(pano_cmd)}")
        r1 = subprocess.run(pano_cmd, capture_output=True, text=True, cwd=REPO_DIR)
        if r1.returncode != 0:
            return None, f"❌ Panorama generation failed:\n{r1.stderr[-3000:]}"

        pano_path = os.path.join(out_path, "panorama.png")
        if not os.path.exists(pano_path):
            return None, f"❌ panorama.png not found after stage 1. Stderr:\n{r1.stderr[-2000:]}"

        # ── Stage 2: Scene generation ─────────────────────────────────────────
        scene_cmd = [
            "python3", f"{REPO_DIR}/demo_scenegen.py",
            "--image_path", pano_path,
            "--classes", scene_class,
            "--output_path", out_path,
        ]
        # Per README: indicate foreground objects to layer out
        if labels_fg1.strip():
            scene_cmd += ["--labels_fg1"] + labels_fg1.strip().split()
        if labels_fg2.strip():
            scene_cmd += ["--labels_fg2"] + labels_fg2.strip().split()

        if use_fp8:
            scene_cmd += ["--fp8_gemm", "--fp8_attention"]

        print(f"Stage 2 command: {' '.join(scene_cmd)}")
        r2 = subprocess.run(scene_cmd, capture_output=True, text=True, cwd=REPO_DIR)
        if r2.returncode != 0:
            return None, f"❌ Scene generation failed:\n{r2.stderr[-3000:]}"

        # Copy modelviewer.html into output so user can open locally
        mv_src = f"{REPO_DIR}/modelviewer.html"
        if os.path.exists(mv_src):
            shutil.copy(mv_src, os.path.join(out_path, "modelviewer.html"))

        # Write a README explaining how to use the viewer
        with open(os.path.join(out_path, "HOW_TO_VIEW.txt"), "w") as f:
            f.write(
                "HOW TO VIEW YOUR 3D WORLD\n"
                "=========================\n\n"
                "1. Extract this ZIP\n"
                "2. Open modelviewer.html in your browser (Chrome recommended)\n"
                "3. Upload the generated 3D scene files when prompted\n"
                "4. Navigate your world in real time!\n\n"
                "Files:\n"
                "  panorama.png      - 360 degree panorama\n"
                "  modelviewer.html  - Browser viewer (open this)\n"
                "  *.glb / *.obj     - 3D mesh files\n"
            )

        # Zip everything
        zip_path = tempfile.mktemp(suffix=".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out_path):
                for fname in files:
                    fp = os.path.join(root, fname)
                    zf.write(fp, os.path.relpath(fp, out_path))

        output_vol.commit()
        return zip_path, (
            "✅ Done!\n"
            "Extract the ZIP, open modelviewer.html in Chrome to walk through your world.\n"
            f"Output: {out_path}"
        )

    # ── UI ────────────────────────────────────────────────────────────────────
    with gr.Blocks(title="HunyuanWorld 1.0") as demo:
        gr.Markdown(
            "# 🌐 HunyuanWorld 1.0\n"
            "**Single image OR text → 360° panorama → explorable 3D world mesh**\n\n"
            "Outputs a ZIP with 3D mesh files + `modelviewer.html` — open it in Chrome to walk around your world."
        )
        with gr.Row():
            with gr.Column():
                mode = gr.Radio(
                    ["Image to World", "Text to World"],
                    value="Image to World",
                    label="Generation Mode",
                )
                img_in = gr.File(
                    label="Input Image (Image to World mode)",
                    file_types=["image"],
                    visible=True,
                )
                prompt_in = gr.Textbox(
                    label="Text Prompt (Text to World mode)",
                    placeholder="A forest with ancient ruins, misty morning light...",
                    lines=3,
                    visible=False,
                )

                # Show/hide based on mode
                def toggle_mode(m):
                    return (
                        gr.update(visible=(m == "Image to World")),
                        gr.update(visible=(m == "Text to World")),
                    )
                mode.change(toggle_mode, inputs=mode, outputs=[img_in, prompt_in])

                gr.Markdown(
                    "### Foreground Object Labels\n"
                    "Tell the model which objects to layer separately (improves scene quality).\n"
                    "Examples: `stones` `trees` `buildings` `sculptures` `flowers` `mountains`"
                )
                with gr.Row():
                    labels_fg1 = gr.Textbox(label="Foreground Layer 1", value="stones",
                                             placeholder="e.g. stones sculptures")
                    labels_fg2 = gr.Textbox(label="Foreground Layer 2", value="trees",
                                             placeholder="e.g. trees mountains")

                scene_class = gr.Dropdown(
                    ["outdoor", "indoor"],
                    value="outdoor",
                    label="Scene Class",
                )
                use_fp8 = gr.Checkbox(
                    value=False,
                    label="FP8 quantization (--fp8_gemm --fp8_attention) — use if OOM",
                )
                run_btn = gr.Button("🌍 Generate World", variant="primary", size="lg")

            with gr.Column():
                status  = gr.Textbox(label="Status", interactive=False, lines=5)
                dl      = gr.File(label="⬇ Download World (ZIP)")
                gr.Markdown(
                    "### After downloading:\n"
                    "1. Extract the ZIP\n"
                    "2. Open **modelviewer.html** in Chrome\n"
                    "3. Walk around your 3D world!"
                )

        run_btn.click(
            run_world,
            inputs=[mode, img_in, prompt_in, labels_fg1, labels_fg2,
                    scene_class, use_fp8],
            outputs=[dl, status],
        )
        gr.Markdown(
            "---\n"
            "**Models:** `tencent/HunyuanWorld-1` (Flux-based) · "
            "**GPU:** A100 40GB · "
            "**Repo:** https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0\n\n"
            "⚠️ First run takes 15–30 min (model download + generation). Subsequent runs ~5–10 min."
        )

    demo.queue(max_size=3)
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
