import asyncio
import os, json, uuid, httpx, aiofiles
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from py.get_setting import DEFAULT_ASR_DIR          # 你的 ASR 目录

router = APIRouter(prefix="/sherpa-model")

MODELS = {
    "modelscope": {
        "url": "https://modelscope.cn/models/pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue/resolve/master/model.int8.onnx",
        "tokens_url": "https://modelscope.cn/models/pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue/resolve/master/tokens.txt",
        "filename": "model.int8.onnx"
    },
    "huggingface": {
        "url": "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/resolve/main/model.int8.onnx?download=true",
        "tokens_url": "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/resolve/main/tokens.txt?download=true",
        "filename": "model.int8.onnx"
    }
}
# ---------- 工具 ----------
def model_exists() -> bool:
    return (Path(DEFAULT_ASR_DIR) / "model.int8.onnx").is_file()

async def download_file(url: str, dest: Path, progress_id: str):
    tmp = dest.with_suffix(".downloading")
    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:  # ← 关键
        async with client.stream("GET", url) as resp:
            total = int(resp.headers.get("content-length", 0))
            done = 0
            async with aiofiles.open(tmp, "wb") as f:
                async for chunk in resp.aiter_bytes(1024 * 64):
                    await f.write(chunk)
                    done += len(chunk)
                    (Path(DEFAULT_ASR_DIR) / f"{progress_id}.json").write_text(
                        json.dumps({"done": done, "total": total})
                    )
    tmp.rename(dest)

# ---------- 接口 ----------
@router.get("/status")
def status():
    return {"exists": model_exists()}

@router.delete("/remove")
def remove():
    onnx = Path(DEFAULT_ASR_DIR) / "model.int8.onnx"
    tokens = Path(DEFAULT_ASR_DIR) / "tokens.txt"
    if onnx.exists():
        onnx.unlink()
    if tokens.exists():
        tokens.unlink()
    return {"ok": True}

@router.get("/download/{source}")
async def download(source: str):
    if source not in MODELS:
        raise HTTPException(status_code=400, detail="bad source")
    if model_exists():
        raise HTTPException(status_code=400, detail="model already exists")
    progress_id = uuid.uuid4().hex
    dest = Path(DEFAULT_ASR_DIR) / MODELS[source]["filename"]

    async def event_generator():
        # 1. 启动模型下载
        asyncio.create_task(
            download_file(MODELS[source]["url"], dest, progress_id)
        )
        # 2. 启动同源的 tokens.txt 下载
        asyncio.create_task(
            download_file(
                MODELS[source]["tokens_url"],
                Path(DEFAULT_ASR_DIR) / "tokens.txt",
                progress_id + "_tok"
            )
        )

        # SSE 推送进度
        while True:
            await asyncio.sleep(0.5)
            try:
                data = json.loads(
                    (Path(DEFAULT_ASR_DIR) / f"{progress_id}.json").read_text()
                )
                yield f"data: {json.dumps(data)}\n\n"
                if data["done"] == data["total"] and data["total"] > 0:
                    (Path(DEFAULT_ASR_DIR) / f"{progress_id}.json").unlink(missing_ok=True)
                    yield "data: close\n\n"
                    return
            except FileNotFoundError:
                yield f"data: {json.dumps({'done': 0, 'total': 0})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
