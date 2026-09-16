#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_server.py — PaddleOCR 逐瓶识别 HTTP 服务

启动：python3 tools/ocr_server.py --host 0.0.0.0 --port 8000
接口：
  GET  /health        → {"status":"ok","engine":"paddleocr"}
  POST /ocr           → multipart file=@photo.jpg
                        响应：{bottle_count, bottles:[{bottle,bbox,name_hint,raw_text,matches:[{id,name,category}]}]}
  POST /ocr_base64    → JSON {"image":"data:image/jpeg;base64,..."}
                        响应同 /ocr

网站前端优先调用此服务，失败时回退浏览器端 Tesseract.js。
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bottle_ocr import (
    load_ocr, run_ocr, cluster_bottles, build_bottle,
    load_aliases, match_brands,
)

app = FastAPI(title="Home Bar Bottle OCR", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_OCR = None
_ALIAS_DB = None


def get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = load_ocr()
    return _OCR


def get_aliases():
    global _ALIAS_DB
    if _ALIAS_DB is None:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "index.html"),
            os.path.join(here, "..", "index.html"),
        ]
        for html in candidates:
            if os.path.exists(html):
                _ALIAS_DB = load_aliases(html)
                break
    return _ALIAS_DB


def process_image(image_bytes: bytes):
    ocr = get_ocr()
    aliases = get_aliases()
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图片")
    img_h, img_w = img.shape[:2]

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        lines = run_ocr(ocr, tmp_path)
    finally:
        os.unlink(tmp_path)

    if not lines:
        return {"bottle_count": 0, "bottles": [], "image_size": {"width": img_w, "height": img_h}}

    groups = cluster_bottles(lines, img_w, img_h)
    bottles = [build_bottle(i + 1, g, lines) for i, g in enumerate(groups)]
    for b in bottles:
        b["matches"] = match_brands(b, aliases)
        b["matched_label"] = "、".join(m["name"] for m in b["matches"]) if b["matches"] else ""

    return {
        "bottle_count": len(bottles),
        "image_size": {"width": img_w, "height": img_h},
        "bottles": bottles,
    }


@app.get("/health")
def health():
    engine_name = "paddleocr" if os.environ.get("OCR_ENGINE", "rapid").lower() == "paddle" else "rapidocr-onnx"
    return {"status": "ok", "engine": engine_name, "aliases_loaded": get_aliases() is not None}


@app.post("/ocr")
async def ocr_upload(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    return process_image(data)


class ImagePayload(BaseModel):
    image: str


@app.post("/ocr_base64")
def ocr_base64(payload: ImagePayload):
    s = payload.image
    if "," in s:
        s = s.split(",", 1)[1]
    try:
        data = base64.b64decode(s)
    except Exception:
        raise HTTPException(status_code=400, detail="base64 解码失败")
    return process_image(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
