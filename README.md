---
title: Bottle OCR
emoji: 🍾
colorFrom: emerald
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Bottle OCR — Home Bar 逐瓶识别服务

基于 PaddleOCR PP-OCRv6 的酒瓶标签逐瓶识别 API，为家庭调酒台网站提供高精度 OCR。

## 接口

- `GET /health` — 健康检查
- `POST /ocr` — multipart 上传图片，返回分瓶识别结果
- `POST /ocr_base64` — JSON body `{"image": "data:image/jpeg;base64,..."}`
