"""
HunyuanWorld 1.0 — Modal Deployment (Stable Build)
====================================================
Repo:   https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
Models: https://huggingface.co/tencent/HunyuanWorld-1 (all public, no token needed)

Strategy:
  - Use pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel as base
    (pre-built PyTorch + CUDA — eliminates the biggest install step)
  - Install sentencepiece + protobuf FIRST (before transformers)
  - Patch basicsr degradations.py immediately after install
  - Install Real-ESRGAN deps in correct order
  - Bake ZIM onnx weights into image (no runtime download)
  - Compile draco once, cached in image layer

Deploy:
  modal deploy world10_modal.py

All parameters from https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0/blob/main/README.md
"""

import modal

app = modal.App("hunyuan-world10")

hf_vol     = modal.Volume.from_name("world10-hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("world10-outputs",  create_if_missing=True)

HF_CACHE   = "/hf_cache"
OUT_DIR    = "/outputs"
REPO_DIR   = "/app/HunyuanWorld-1.0"
ZIM_DIR    = "/app/ZIM"
ESRGAN_DIR = "/app/Real-ESRGAN"

# ── Patch script — fixes basicsr broken import, baked into image ──────────────
# Uses site.getsitepackages() so it works on any Python version (3.10 or 3.11)
BASICSR_PATCH = (
    "import site, os; "
    "sp = site.getsitepackages()[0]; "
    "f = os.path.join(sp, 'basicsr/data/degradations.py'); "
    "txt = open(f).read(); "
    "fixed = txt.replace("
    "    'from torchvision.transforms.functional_tensor import rgb_to_grayscale',"
    "    'from torchvision.transforms.functional import rgb_to_grayscale'"
    "); "
    "open(f, 'w').write(fixed); "
    "print('basicsr patch OK')"
)

# ── Container Image ───────────────────────────────────────────────────────────
# Base: pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel
#   → PyTorch 2.5.0 + CUDA 12.4 pre-installed — no need to build from bare CUDA
# ─────────────────────────────────────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel",
    )
    .apt_install(
        "git", "wget",
        "libgl1", "libglib2.0-0", "libgomp1",
        "cmake", "build-essential", "ninja-build",
        "ffmpeg",
    )

    # ── Step 1: sentencepiece + protobuf BEFORE everything else ──────────────
    # Must precede transformers — T5 fast tokenizer checks for sentencepiece
    # at import time. If missing: "Cannot instantiate this tokenizer from a slow version"
    .pip_install("sentencepiece", "protobuf")

    # ── Step 2: Core ML stack ─────────────────────────────────────────────────
    .pip_install(
        "diffusers==0.31.0",
        "transformers==4.46.3",
        "accelerate",
        "tokenizers",
        "huggingface_hub",
        "safetensors",
    )

    # ── Step 3: basicsr + Real-ESRGAN ─────────────────────────────────────────
    # gfpgan pulls in unpatched basicsr — patch immediately after, and again
    # after Real-ESRGAN setup.py develop which may reinstall basicsr
    .pip_install("basicsr-fixed", "facexlib", "gfpgan")
    .run_commands(f'python3 -c "{BASICSR_PATCH}"')
    .run_commands(
        f"git clone https://github.com/xinntao/Real-ESRGAN.git {ESRGAN_DIR}",
        f"cd {ESRGAN_DIR} && pip install -r requirements.txt --no-deps",
        f"cd {ESRGAN_DIR} && python setup.py develop",
        f'python3 -c "{BASICSR_PATCH}"',
    )

    # ── Step 4: ZIM segmentation — bake ONNX weights into image ──────────────
    .run_commands(
        f"git clone https://github.com/naver-ai/ZIM.git {ZIM_DIR}",
        f"cd {ZIM_DIR} && pip install -e . --no-deps",
        f"mkdir -p {ZIM_DIR}/zim_vit_l_2092",
        f"wget -q https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx -O {ZIM_DIR}/zim_vit_l_2092/encoder.onnx",
        f"wget -q https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx -O {ZIM_DIR}/zim_vit_l_2092/decoder.onnx",
    )

    # ── Step 5: draco mesh compiler (Ninja is faster than plain make) ─────────
    .run_commands(
        "git clone --depth 1 https://github.com/google/draco.git /tmp/draco",
        "cd /tmp/draco && cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release .",
        "cd /tmp/draco && ninja -C build && ninja -C build install",
        "rm -rf /tmp/draco",
    )

    # ── Step 6: Remaining Python deps ─────────────────────────────────────────
    .pip_install(
        "einops",
        "omegaconf",
        "scipy",
        "trimesh",
        "Pillow",
        "opencv-python",
        "numpy<2.0.0",
        "tqdm",
        "open3d==0.18.0",
        "onnxruntime-gpu",
        # gradio==5.49.1 — 5.9.1 has a known 500 Server Error schema bug
        "gradio==5.49.1",
        "fastapi[standard]",
    )

    # ── Step 7: Clone HunyuanWorld-1.0 repo ──────────────────────────────────
    .run_commands(
        f"git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git {REPO_DIR}",
    )

    .env({
        "PYTHONPATH": REPO_DIR,
        "HF_HOME":    HF_CACHE,
        "ZIM_MODEL_PATH": f"{ZIM_DIR}/zim_vit_l_2092",
    })
)


# ── Gradio App ────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A100",
    timeout=2400,
    volumes={
        HF_CACHE:  hf_vol,
        OUT_DIR:   output_vol,
    },
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

    def run_world(mode, image_file, prompt, labels_fg1, labels_fg2,
                  scene_class, use_fp8, use_cache):
        out_path = tempfile.mkdtemp(dir=OUT_DIR, prefix="hw10_")

        # Stage 1 — Panorama
        pano_cmd = ["python3", f"{REPO_DIR}/demo_panogen.py",
                    "--output_path", out_path]

        if mode == "Image to World":
            if image_file is None:
                return None, "❌ Upload an image."
            # Gradio 5.x File returns FileData with .path, not .name
            img_src = getattr(image_file, 'path', None) or getattr(image_file, 'name', None) or str(image_file)
            img_dst = os.path.join(out_path, "input.png")
            shutil.copy(img_src, img_dst)
            pano_cmd += ["--prompt", "", "--image_path", img_dst]
        else:
            if not prompt.strip():
                return None, "❌ Enter a text prompt."
            pano_cmd += ["--prompt", prompt.strip()]

        if use_fp8:
            pano_cmd += ["--fp8_gemm", "--fp8_attention"]
        if use_cache:
            pano_cmd += ["--cache"]

        r1 = subprocess.run(pano_cmd, capture_output=True, text=True, cwd=REPO_DIR)
        if r1.returncode != 0:
            return None, f"❌ Panorama failed (stage 1):\n{r1.stderr[-3000:]}"

        pano_path = os.path.join(out_path, "panorama.png")
        if not os.path.exists(pano_path):
            return None, (
                f"❌ panorama.png missing.\n"
                f"stdout: {r1.stdout[-1000:]}\nstderr: {r1.stderr[-1000:]}"
            )

        # Stage 2 — Scene generation
        scene_cmd = ["python3", f"{REPO_DIR}/demo_scenegen.py",
                     "--image_path", pano_path,
                     "--classes", scene_class,
                     "--output_path", out_path]

        if labels_fg1.strip():
            scene_cmd += ["--labels_fg1"] + labels_fg1.strip().split()
        if labels_fg2.strip():
            scene_cmd += ["--labels_fg2"] + labels_fg2.strip().split()
        if use_fp8:
            scene_cmd += ["--fp8_gemm", "--fp8_attention"]
        if use_cache:
            scene_cmd += ["--cache"]

        r2 = subprocess.run(scene_cmd, capture_output=True, text=True, cwd=REPO_DIR)
        if r2.returncode != 0:
            return None, f"❌ Scene generation failed (stage 2):\n{r2.stderr[-3000:]}"

        # Pack outputs
        mv_html = f"{REPO_DIR}/modelviewer.html"
        if os.path.exists(mv_html):
            shutil.copy(mv_html, os.path.join(out_path, "modelviewer.html"))

        with open(os.path.join(out_path, "HOW_TO_VIEW.txt"), "w") as f:
            f.write(
                "HOW TO VIEW YOUR 3D WORLD\n"
                "=========================\n"
                "1. Extract this ZIP\n"
                "2. Open modelviewer.html in Chrome\n"
                "3. Upload the scene files when prompted\n"
                "4. Walk around your world!\n\n"
                "Files:\n"
                "  panorama.png     - 360 panorama\n"
                "  modelviewer.html - browser viewer\n"
                "  *.glb / *.obj    - 3D mesh (import into Blender/Unity/UE)\n"
            )

        zip_path = tempfile.mktemp(suffix=".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out_path):
                for fname in files:
                    fp = os.path.join(root, fname)
                    zf.write(fp, os.path.relpath(fp, out_path))

        output_vol.commit()
        return zip_path, "✅ Done! Extract ZIP → open modelviewer.html in Chrome."

    # ── UI ────────────────────────────────────────────────────────────────────
    with gr.Blocks(title="HunyuanWorld 1.0") as demo:
        gr.Markdown(
            "# 🌐 HunyuanWorld 1.0\n"
            "**Single image OR text → 360° panorama → explorable 3D world**\n\n"
            "Outputs a ZIP with 3D mesh files + `modelviewer.html`.\n"
            "Open `modelviewer.html` in **Chrome** to walk around your world."
        )
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    choices=["Image to World", "Text to World"],
                    value="Image to World",
                    label="Mode",
                )
                img_in = gr.File(
                    label="Input Image",
                    file_types=["image"],
                    visible=True,
                )
                prompt_in = gr.Textbox(
                    label="Text Prompt",
                    placeholder="A forest with ancient ruins at sunset...",
                    lines=3,
                    visible=False,
                )
                def toggle(m):
                    return (
                        gr.update(visible=(m == "Image to World")),
                        gr.update(visible=(m == "Text to World")),
                    )
                mode.change(toggle, inputs=mode, outputs=[img_in, prompt_in])

                gr.Markdown(
                    "### Foreground Labels\n"
                    "Objects to layer separately. Examples: `stones trees buildings flowers`"
                )
                with gr.Row():
                    fg1 = gr.Textbox(label="Layer 1", value="stones")
                    fg2 = gr.Textbox(label="Layer 2", value="trees")

                scene_class = gr.Dropdown(
                    choices=["outdoor", "indoor"],
                    value="outdoor",
                    label="Scene Class",
                )
                with gr.Row():
                    use_fp8   = gr.Checkbox(value=False, label="FP8 — save VRAM")
                    use_cache = gr.Checkbox(value=False, label="Cache — faster")

                run_btn = gr.Button("🌍 Generate World", variant="primary", size="lg")

            with gr.Column(scale=1):
                status = gr.Textbox(label="Status", interactive=False, lines=6)
                dl     = gr.File(label="⬇ Download World ZIP")
                gr.Markdown(
                    "**After download:**\n"
                    "1. Extract ZIP\n"
                    "2. Open **modelviewer.html** in Chrome\n"
                    "3. Walk around your world 🎮"
                )

        run_btn.click(
            run_world,
            inputs=[mode, img_in, prompt_in, fg1, fg2, scene_class, use_fp8, use_cache],
            outputs=[dl, status],
        )
        gr.Markdown(
            "---\n"
            "**Models:** `tencent/HunyuanWorld-1` · "
            "**GPU:** A100 40GB · "
            "**Source:** https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0\n\n"
            "⚠️ First generation: ~20 min (model download + panorama + scene). "
            "Subsequent runs: ~15 min (weights cached)."
        )

    demo.queue(max_size=3)
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
