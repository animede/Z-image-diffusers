#!/usr/bin/env python3
"""Z-Image-Turbo Gradio デモ UI。

このUIはモデルを直接ロードしません。すべての生成は httpx 経由で
api_server.py (デフォルト http://localhost:8000) を叩くだけのクライアントです。
先に `venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000` を起動してください。

3タブ構成:
  - T2I            : テキストから画像生成 (ZImagePipeline)
  - I2I            : 参照画像 + プロンプトで画像生成 (ZImageImg2ImgPipeline)
  - Edit(インペイント): マスクで指定した領域だけを再生成 (ZImageInpaintPipeline)
                       ※ 公式 Z-Image-Edit チェックポイントは未公開のため、
                       このプロジェクトでは同じ Z-Image-Turbo 重みを使った
                       マスクインペイントとして実装しています。
"""
import base64
import io
import os

import gradio as gr
import httpx
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# 生成は数秒〜(初回は追加でモデルロード/ダウンロードに数分)かかるため長めに設定
_client = httpx.Client(base_url=API_URL, timeout=httpx.Timeout(600.0, connect=10.0))


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_to_b64(mask: Image.Image) -> str:
    buf = io.BytesIO()
    mask.convert("L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_img(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _fetch_health():
    try:
        r = _client.get("/health")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "model_loaded": False, "config": {}, "error": str(e)}


def _format_info(payload, extra=""):
    seed = payload.get("seed")
    elapsed = payload.get("elapsed_sec", 0.0)
    vram = payload.get("peak_vram_gb", 0.0)
    paths = payload.get("saved_paths", [])
    lines = [
        f"生成時間: {elapsed:.2f} 秒 / 最大VRAM使用量: {vram:.2f} GB / 使用シード: {seed}",
        f"保存先: {', '.join(paths)}",
    ]
    if extra:
        lines.insert(0, extra)
    return "\n".join(lines)


def _post(path, payload):
    try:
        r = _client.post(path, json=payload)
    except httpx.RequestError as e:
        raise gr.Error(f"APIサーバに接続できません ({API_URL}): {type(e).__name__}: {e}")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise gr.Error(f"生成に失敗しました (HTTP {r.status_code}): {detail}")
    return r.json()


def generate_t2i(prompt, negative_prompt, steps, guidance, width, height, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "steps": int(steps),
        "guidance_scale": float(guidance),
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "num_images": 1,
    }
    data = _post("/api/t2i", payload)
    image = _b64_to_img(data["images"][0])
    return image, _format_info(data)


def generate_i2i(input_image, prompt, negative_prompt, strength, steps, guidance, width, height, seed):
    if input_image is None:
        raise gr.Error("入力画像をアップロードしてください。")
    if not prompt or not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "steps": int(steps),
        "guidance_scale": float(guidance),
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "num_images": 1,
        "image": _img_to_b64(input_image),
        "strength": float(strength),
    }
    data = _post("/api/i2i", payload)
    image = _b64_to_img(data["images"][0])
    return image, _format_info(data)


def _extract_background_and_mask(editor_value):
    """gr.ImageMask (gr.ImageEditor ベース) の EditorValue からベース画像とマスクを取り出す。

    ImageMask は brush=黒固定・layers=単一。ペイントしたレイヤーの
    アルファチャンネル(不透明=描画箇所)を二値化してマスク(白=編集領域)とする。
    """
    if editor_value is None:
        return None, None
    background = editor_value.get("background")
    layers = editor_value.get("layers") or []
    if background is None:
        return None, None
    background = background.convert("RGB")
    if not layers or layers[0] is None:
        return background, None
    layer = layers[0]
    if layer.mode != "RGBA":
        layer = layer.convert("RGBA")
    alpha = layer.split()[-1]
    mask = alpha.point(lambda p: 255 if p > 10 else 0).convert("L")
    if mask.getbbox() is None:
        return background, None
    return background, mask


def generate_edit(editor_value, prompt, negative_prompt, strength, steps, guidance, width, height, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    background, mask = _extract_background_and_mask(editor_value)
    if background is None:
        raise gr.Error("画像をアップロードしてください。")
    if mask is None:
        raise gr.Error("マスク(編集したい領域)をブラシで塗ってください。")

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "steps": int(steps),
        "guidance_scale": float(guidance),
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "num_images": 1,
        "image": _img_to_b64(background),
        "mask": _mask_to_b64(mask),
        "strength": float(strength),
    }
    data = _post("/api/edit", payload)
    image = _b64_to_img(data["images"][0])
    return image, _format_info(data)


_health = _fetch_health()
_config = _health.get("config", {}) or {}
_header_md = (
    f"# Z-Image-Turbo デモ UI\n\n"
    f"API サーバ: `{API_URL}` "
    f"(状態: **{_health.get('status')}**, モデルロード済み: **{_health.get('model_loaded')}**)\n\n"
    f"設定: precision=`{_config.get('precision', '?')}` / offload=`{_config.get('offload', '?')}` / "
    f"attn=`{_config.get('attention_backend', '?')}` / compile=`{_config.get('compile', '?')}` / "
    f"vae_tiling=`{_config.get('vae_tiling', '?')}`\n\n"
    "Z-Image-Turbo は蒸留モデルのため既定は8ステップ・ガイダンス0.0(CFGなし)です。"
    "「Edit」タブは公式 Z-Image-Edit 未公開のため、マスクインペイントとして実装しています。"
)

with gr.Blocks(title="Z-Image-Turbo デモ") as demo:
    gr.Markdown(_header_md)

    with gr.Tabs():
        with gr.TabItem("T2I(テキストから画像生成)"):
            with gr.Row():
                with gr.Column():
                    t2i_prompt = gr.Textbox(
                        label="プロンプト", lines=4,
                        placeholder="生成したい画像の説明を入力してください",
                    )
                    t2i_negative = gr.Textbox(
                        label="ネガティブプロンプト(guidance_scale>0の時のみ有効)", lines=2,
                    )
                    with gr.Row():
                        t2i_steps = gr.Slider(1, 50, value=8, step=1, label="ステップ数")
                        t2i_guidance = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="ガイダンススケール")
                    with gr.Row():
                        t2i_width = gr.Slider(512, 2048, value=1024, step=64, label="幅")
                        t2i_height = gr.Slider(512, 2048, value=1024, step=64, label="高さ")
                    t2i_seed = gr.Number(value=-1, label="シード(-1でランダム)", precision=0)
                    t2i_btn = gr.Button("画像を生成", variant="primary")
                with gr.Column():
                    t2i_output = gr.Image(label="生成結果")
                    t2i_info = gr.Textbox(label="生成情報", interactive=False, lines=2)
            t2i_btn.click(
                generate_t2i,
                inputs=[t2i_prompt, t2i_negative, t2i_steps, t2i_guidance, t2i_width, t2i_height, t2i_seed],
                outputs=[t2i_output, t2i_info],
            )

        with gr.TabItem("I2I(画像から画像生成)"):
            with gr.Row():
                with gr.Column():
                    i2i_image = gr.Image(label="入力画像", type="pil")
                    i2i_prompt = gr.Textbox(
                        label="プロンプト", lines=4,
                        placeholder="どのように変化させたいか説明してください",
                    )
                    i2i_negative = gr.Textbox(
                        label="ネガティブプロンプト(guidance_scale>0の時のみ有効)", lines=2,
                    )
                    i2i_strength = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="strength(入力画像からの変化度)")
                    with gr.Row():
                        i2i_steps = gr.Slider(1, 50, value=8, step=1, label="ステップ数")
                        i2i_guidance = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="ガイダンススケール")
                    with gr.Row():
                        i2i_width = gr.Slider(512, 2048, value=1024, step=64, label="幅")
                        i2i_height = gr.Slider(512, 2048, value=1024, step=64, label="高さ")
                    i2i_seed = gr.Number(value=-1, label="シード(-1でランダム)", precision=0)
                    i2i_btn = gr.Button("画像を生成", variant="primary")
                with gr.Column():
                    i2i_output = gr.Image(label="生成結果")
                    i2i_info = gr.Textbox(label="生成情報", interactive=False, lines=2)
            i2i_btn.click(
                generate_i2i,
                inputs=[
                    i2i_image, i2i_prompt, i2i_negative, i2i_strength,
                    i2i_steps, i2i_guidance, i2i_width, i2i_height, i2i_seed,
                ],
                outputs=[i2i_output, i2i_info],
            )

        with gr.TabItem("Edit(インペイント)"):
            gr.Markdown(
                "画像をアップロードし、ブラシで**編集したい領域を塗って**からプロンプトを入力してください。"
                "公式 Z-Image-Edit モデルは未公開のため、Z-Image-Turbo の重みを用いた"
                "マスクインペイントとして実装しています。"
            )
            with gr.Row():
                with gr.Column():
                    edit_editor = gr.ImageMask(label="画像 + マスク(ブラシで編集領域を塗る)", type="pil")
                    edit_prompt = gr.Textbox(
                        label="プロンプト", lines=4,
                        placeholder="塗った領域に何を生成するか説明してください",
                    )
                    edit_negative = gr.Textbox(
                        label="ネガティブプロンプト(guidance_scale>0の時のみ有効)", lines=2,
                    )
                    edit_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="strength(マスク領域の変化度)")
                    with gr.Row():
                        edit_steps = gr.Slider(1, 50, value=8, step=1, label="ステップ数")
                        edit_guidance = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="ガイダンススケール")
                    with gr.Row():
                        edit_width = gr.Slider(512, 2048, value=1024, step=64, label="幅")
                        edit_height = gr.Slider(512, 2048, value=1024, step=64, label="高さ")
                    edit_seed = gr.Number(value=-1, label="シード(-1でランダム)", precision=0)
                    edit_btn = gr.Button("画像を生成", variant="primary")
                with gr.Column():
                    edit_output = gr.Image(label="生成結果")
                    edit_info = gr.Textbox(label="生成情報", interactive=False, lines=2)
            edit_btn.click(
                generate_edit,
                inputs=[
                    edit_editor, edit_prompt, edit_negative, edit_strength,
                    edit_steps, edit_guidance, edit_width, edit_height, edit_seed,
                ],
                outputs=[edit_output, edit_info],
            )

demo.queue(max_size=10, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862)
