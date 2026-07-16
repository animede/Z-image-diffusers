# Z-Image-Turbo diffusers デモワークスペース

`Tongyi-MAI/Z-Image-Turbo` を [diffusers](https://github.com/huggingface/diffusers) で動かす
「APIサーバ + Gradio UI」実装です。設計スタイルは `~/flux2_diffusers` を踏襲しています
(docstring による設計意図の説明、レジストリパターン、フォールバック処理、日本語UI)。

Z-Image-Turbo は蒸留(ディスティル)モデルで、**8ステップ・guidance_scale=0.0(CFGなし)**が
既定の生成設定です。`guidance_scale` を1超にすると `negative_prompt` が効くようになりますが、
transformer のフォワードパスが2倍(条件付き+無条件)になり生成が約2倍遅くなります。

**Edit(編集)機能について**: 2026年7月現在、Tongyi-MAI は公式の Z-Image-Edit チェックポイントを
公開していません。そのため本プロジェクトの「Edit」は、同じ Z-Image-Turbo の重みを使った
`ZImageInpaintPipeline` によるマスクインペイント(マスクで指定した領域だけを再生成)として
実装しています。公式 Edit モデルが公開された際は `pipeline_manager.py` にモデルIDと
ビルダー関数を追加するだけで切り替えられるよう、レジストリパターンで構成しています。

## 環境

- ワークスペース: `~/Z-image-diffusers/`
- ベース Python: `~/comfy-env/bin/python3`
  (torch 2.9.0+cu128 / transformers 5.1.0 / accelerate / bitsandbytes 動作確認済み。
  他プロジェクトと共有しているため、**comfy-env に直接 pip install しない**)
- GPU: RTX PRO 6000 Blackwell 96GB VRAM (sm_120)
- RAM: 62GB / ディスク空き: 約52GB(タイト。ダウンロードするモデルは
  `Tongyi-MAI/Z-Image-Turbo` のみ、約20GB)

### comfy-env の diffusers ではなぜ動かないか

comfy-env の diffusers は 0.36.0 で `ZImageInpaintPipeline` が存在しません。
このワークスペース専用の `venv/` に diffusers 0.39.0 をインストールし、
`--system-site-packages` で継承した comfy-env の 0.36.0 を **shadow(上書き)** させることで
解決しています(venv 側の site-packages が優先されるため、torch 等はそのまま comfy-env から
継承しつつ diffusers だけを新しくできる)。

## セットアップ

venv はセットアップ済みです(再作成する場合):

```bash
cd ~/Z-image-diffusers
~/comfy-env/bin/python3 -m venv --system-site-packages venv
echo ~/comfy-env/lib/python3.12/site-packages > venv/lib/python3.12/site-packages/comfy_env.pth
venv/bin/pip install "diffusers==0.39.0" fastapi "uvicorn[standard]" python-multipart httpx "gradio>=6.0"
```

検証:

```bash
venv/bin/python -c "import diffusers, torch; from diffusers import ZImagePipeline, ZImageImg2ImgPipeline, ZImageInpaintPipeline; print(diffusers.__version__, diffusers.__file__)"
# -> 0.39.0 と、パスが venv/lib/python3.12/site-packages/diffusers/__init__.py であること
```

モデルは初回実行時に Hugging Face Hub からダウンロードされます(`Tongyi-MAI/Z-Image-Turbo`、
約20GB、ゲート不要)。

## pipeline_manager.py: 設定(環境変数)

`ZImageConfig` が環境変数で切替可能です。

| 環境変数 | デフォルト | 説明 |
|---|---|---|
| `ZIMAGE_PRECISION` | `bf16` | `bf16`(既定、フル精度、約20GB、96GB GPUなら最速)/ `bnb-4bit`(bitsandbytes による transformer + text_encoder のロード時4bit量子化、低VRAM向け) |
| `ZIMAGE_OFFLOAD` | `none` | `none`(既定、全コンポーネントGPU常駐、最速、約21GB)/ `model`(`enable_model_cpu_offload()`、1コンポーネントずつGPU常駐)/ `group`(`group_offload` leaf_level + CUDAストリーム、最小VRAM) |
| `ZIMAGE_ATTN` | `default` | transformer の attention backend。`default`(=SDPA)/ `native` / `_native_flash` / `_native_cudnn` / `_native_efficient` / `flex` / `xformers`。利用不可の backend は警告を出して SDPA にフォールバック |
| `ZIMAGE_COMPILE` | `0` | `1` で transformer を torch.compile(`compile_repeated_blocks` によるリージョナルコンパイル。失敗時は通常の torch.compile にフォールバック) |
| `ZIMAGE_VAE_TILING` | `0` | `1` で VAE の `enable_slicing()` + `enable_tiling()`(高解像度生成・低VRAM decode向け) |
| `ZIMAGE_DEVICE` | `cuda` | offload=`none` の場合の配置先デバイス |
| `ZIMAGE_GROUP_STREAM` | `1` | `ZIMAGE_OFFLOAD=group` の場合の CUDA ストリームプリフェッチ有無 |

既定値(`ZIMAGE_OFFLOAD=none` + `ZIMAGE_PRECISION=bf16`)は **96GB VRAM のこのマシン向けに
全コンポーネントGPU常駐・最速構成**です。低VRAM環境向けの選択肢:

```bash
# VRAM節約(bitsandbytes 4bit + model offload)
ZIMAGE_PRECISION=bnb-4bit ZIMAGE_OFFLOAD=model venv/bin/python test_t2i.py --prompt "..."

# さらに VRAM節約(group offload、8GB クラス GPU でも動作見込み)
ZIMAGE_PRECISION=bnb-4bit ZIMAGE_OFFLOAD=group venv/bin/python test_t2i.py --prompt "..."
```

sm_120(Blackwell)固有の既知問題:

- `sage` / `_sage_*` (sageattention): インストール済みビルドに sm_120 カーネルが無く、
  カーネル起動時に非同期 CUDA エラーになる。`KNOWN_BAD_ATTN_BACKEND_PREFIXES` により
  起動前に拒否し、SDPA にフォールバックする。
- `flash` / `flash_varlen` (flash-attn): パッケージ未インストール。
- 動作確認済み: `default` / `native` / `_native_flash` / `_native_cudnn` /
  `_native_efficient` / `flex` / `xformers`。

## テスト(CLI)

`pipeline_manager.py` を直接使う単体スクリプトです。`outputs/` に画像を保存し、
生成時間と `torch.cuda.max_memory_allocated()` によるピークVRAMを表示します。

```bash
cd ~/Z-image-diffusers

# T2I
venv/bin/python test_t2i.py --prompt "A photorealistic portrait of a red fox wearing a wizard hat" \
  --steps 8 --guidance 0.0 --width 1024 --height 1024

# I2I(strengthで入力画像からの変化度を指定)
venv/bin/python test_i2i.py --image outputs/t2i_xxx.png \
  --prompt "Turn the scene into a vibrant watercolor painting" --strength 0.6

# Edit(インペイント。--image/--maskを省略すると合成テスト画像+矩形マスクを自動生成)
venv/bin/python test_edit.py --prompt "A glowing blue crystal formation" --strength 1.0
```

## api_server.py (FastAPI, port 8000)

起動:

```bash
venv/bin/python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

(`uvicorn` は comfy-env に既存インストール済みのため、`--system-site-packages` venv では
`venv/bin/uvicorn` という実行スクリプトは生成されません。`python -m uvicorn` で起動してください。
ポート8000が他プロセスで使用中の場合は `--port` を変更してください)

モデルは初回リクエスト時に遅延ロードされます。GPU 生成は `threading.Lock` で直列化されます。

### エンドポイント

- `GET /health` -> `{"status": "ok", "model_loaded": bool, "config": {...}}`
- `POST /api/t2i` -> T2I(ZImagePipeline)
- `POST /api/i2i` -> I2I(ZImageImg2ImgPipeline)
- `POST /api/edit` -> Edit / インペイント(ZImageInpaintPipeline)

共通レスポンス: `{"images": [base64 png, ...], "seed": int, "elapsed_sec": float, "peak_vram_gb": float, "saved_paths": [...]}`
(画像は `outputs/` にも保存されます)

### リクエストボディ

`/api/t2i`:

```json
{
  "prompt": "...",
  "negative_prompt": "",
  "steps": 8,
  "guidance_scale": 0.0,
  "width": 1024,
  "height": 1024,
  "seed": -1,
  "num_images": 1
}
```

`/api/i2i`: 上記 + `{"image": "<base64 PNG/JPEG>", "strength": 0.6}`

`/api/edit`: 上記 + `{"image": "<base64>", "mask": "<base64、白=編集領域>", "strength": 1.0}`

### curl 例

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/api/t2i \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A photorealistic portrait of a red fox wearing a wizard hat", "steps": 8, "guidance_scale": 0.0, "width": 1024, "height": 1024, "seed": 42}'

# I2I / Edit は画像をbase64化して渡す
IMG_B64=$(base64 -w0 outputs/t2i_xxx.png)
curl -X POST http://localhost:8000/api/i2i \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Turn the scene into a watercolor painting\", \"image\": \"$IMG_B64\", \"strength\": 0.6}"
```

## ui.py (Gradio, port 7862)

**API サーバを httpx で叩くだけのクライアント**です(パイプラインを直接ロードしません)。
先に `api_server.py` を起動してください。

```bash
API_URL=http://localhost:8000 venv/bin/python ui.py
```

- ポート: `7862`(`http://<ホスト>:7862/`)
- ヘッダーに `/health` の内容(APIサーバの状態・現在の設定)を表示
- 3タブ構成:
  - **T2I**: プロンプト・ネガティブ・ステップ数(1-50、既定8)・ガイダンス(0-10、既定0.0)・
    幅/高さ(512-2048、64刻み、既定1024)・シード
  - **I2I**: 入力画像 + strength(0-1、既定0.6)+ 上記パラメータ
  - **Edit(インペイント)**: `gr.ImageMask` で画像アップロード + ブラシでマスク描画 + プロンプト +
    strength(既定1.0)。マスクはペイントレイヤーのアルファチャンネルを二値化して抽出
- 各タブに生成時間 / 最大VRAM / シード / 保存先を表示

## ディレクトリ構成

```
Z-image-diffusers/
├── venv/                  # このワークスペース専用(diffusers 0.39.0 で comfy-env の
│                            0.36.0 を shadow、他は --system-site-packages で継承)
├── pipeline_manager.py     # パイプラインのシングルトンロード・精度/最適化切替
├── api_server.py           # FastAPI サーバ(T2I/I2I/Edit、ポート8000)
├── ui.py                   # Gradio UI(httpxクライアント、ポート7862)
├── test_t2i.py             # CLI: T2I テスト
├── test_i2i.py             # CLI: I2I テスト
├── test_edit.py            # CLI: Edit(インペイント)テスト(合成マスク自動生成対応)
├── requirements.txt        # venv に追加インストールするパッケージ
├── outputs/                # 生成画像の保存先
└── README.md
```

## 既知の問題・申し送り

- sm_120 対応の sageattention / flash-attn を導入すればさらなる高速化余地があります。
  導入後は `pipeline_manager.py` の `KNOWN_BAD_ATTN_BACKEND_PREFIXES` から `sage` を
  外してください。
- 公式 Z-Image-Edit チェックポイントが公開された場合は、`pipeline_manager.py` に
  新しいモデルID・ビルダー関数を追加して `get_inpaint_pipeline` 相当の切り替えを
  行ってください(現状はインペイントで代替)。
- ディスク空きが約52GBとタイトなため、追加モデルのダウンロード時は空き容量に注意してください。

## ライセンス

このプロジェクトのソースコードは MIT ライク(参照実装 `~/flux2_diffusers` と
同様の方針)で構成しています。利用するモデル(`Tongyi-MAI/Z-Image-Turbo`)や依存ライブラリ
には、それぞれのライセンスが別途適用されます。
