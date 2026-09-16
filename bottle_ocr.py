#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bottle_ocr.py — 多酒瓶照片逐瓶识别（分瓶 -> 逐瓶 OCR -> 品牌匹配）

流程：
  1. PaddleOCR (PP-OCRv6, 中英文) 检测图中所有文字行及坐标；
  2. 按文字框的空间位置聚类：先按 x 轴区间连通分量区分并排酒瓶，
     再在每个簇内按 y 轴大间隙切分前后叠放的酒瓶；
  3. 每瓶独立输出：瓶号、bbox、逐行文字（含置信度）、候选酒名；
  4. 用网站 index.html 中的 ALIASES 品牌别名库逐瓶匹配库存原料。

用法：
  python3 bottle_ocr.py 照片.jpg                 # 输出 JSON
  python3 bottle_ocr.py 照片.jpg --vis out.png   # 额外输出画框可视化图
  python3 bottle_ocr.py 照片.jpg --pretty        # 人类可读格式
  python3 bottle_ocr.py 照片.jpg --html ../index.html
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

NOISE_PATTERNS = [
    r"%?\s*vol\b", r"\b\d{2,3}\s*ml\b", r"\b\d+(\.\d+)?\s*cl\b",
    r"酒精度", r"净含量", r"配料", r"原料", r"产品标准", r"生产许可",
    r"生产日期", r"保质期", r"贮存", r"产地", r"地址", r"电话",
    r"400[-\s]?\d{3}[-\s]?\d{4}", r"www\.", r"\.com", r"\.cn",
    r"e\s*\d{3}", r"^\d{6,}$", r"imported", r"product of",
    r"^[\d\s.,%xX°度volmLcL/]+$",
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


MAX_SIDE = int(os.environ.get("OCR_MAX_SIDE", "1280"))
ENGINE = os.environ.get("OCR_ENGINE", "rapid").lower()


def load_ocr():
    if ENGINE == "paddle":
        from paddleocr import PaddleOCR
        return ("paddle", PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="ch",
        ))
    from rapidocr_onnxruntime import RapidOCR
    return ("rapid", RapidOCR())


def _load_image(image_path):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图片: " + image_path)
    h, w = img.shape[:2]
    scale = 1.0
    longest = max(h, w)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / float(longest)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def run_ocr(engine, image_path):
    """返回 [{text, score, box:[x1,y1,x2,y2]}]，已过滤低置信度行。坐标为原图坐标。"""
    kind, ocr = engine
    import numpy as np
    img, scale = _load_image(image_path)

    if kind == "paddle":
        result = ocr.predict(img)
        r = result[0]
        texts = list(r["rec_texts"])
        scores = list(r["rec_scores"])
        boxes = list(r["rec_boxes"])
        raw = []
        for text, score, box in zip(texts, scores, boxes):
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            raw.append((text.strip(), float(score), [x1, y1, x2, y2]))
    else:
        result, _ = ocr(img)
        raw = []
        if result:
            for box, text, score in result:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                raw.append((text.strip(), float(score), [min(xs), min(ys), max(xs), max(ys)]))

    inv = 1.0 / scale if scale else 1.0
    lines = []
    for text, score, box in raw:
        x1, y1, x2, y2 = [int(round(v * inv)) for v in box]
        lines.append({"text": text, "score": round(score, 4), "box": [x1, y1, x2, y2]})
    return [ln for ln in lines if ln["text"] and ln["score"] >= 0.45]


def x_gap(a, b):
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    if ax2 < bx1:
        return bx1 - ax2
    if bx2 < ax1:
        return ax1 - bx2
    return 0


def cluster_bottles(lines, img_w, img_h):
    """
    分瓶：
      A. x 轴区间连通分量（并排瓶）；
      B. 簇内 y 轴大间隙切分（前后叠放/上下两瓶）。
    返回组列表，每组是 line 的下标数组，按 (瓶左->右, 上->下) 排序。
    """
    boxes = [ln["box"] for ln in lines]
    heights = [b[3] - b[1] for b in boxes]
    med_h = max(median(heights), 12)
    gap_x = max(med_h * 1.2, img_w * 0.01)

    # A. 并查集：x 区间重叠或间隙 <= gap_x 即同瓶
    parent = list(range(len(lines)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if x_gap(boxes[i], boxes[j]) <= gap_x:
                # 只在 y 投影也接近时连通：避免对角线误连（间隙还要小于 2.5 倍行高）
                yi = (boxes[i][1] + boxes[i][3]) / 2
                yj = (boxes[j][1] + boxes[j][3]) / 2
                if abs(yi - yj) < med_h * 6:
                    union(i, j)

    groups = {}
    for i in range(len(lines)):
        groups.setdefault(find(i), []).append(i)

    # B. 每组内按 y 大间隙拆分（前后叠放的两瓶 x 区间可能重叠）
    final_groups = []
    gap_y = med_h * 3.5
    for idxs in groups.values():
        idxs_sorted = sorted(idxs, key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
        start = 0
        for k in range(1, len(idxs_sorted)):
            prev_bottom = boxes[idxs_sorted[k - 1]][3]
            cur_top = boxes[idxs_sorted[k]][1]
            if cur_top - prev_bottom > gap_y:
                final_groups.append(idxs_sorted[start:k])
                start = k
        final_groups.append(idxs_sorted[start:])

    # 组排序：从左到右（cx），同列从上到下（cy）
    def group_key2(g):
        cx = sum((boxes[i][0] + boxes[i][2]) for i in g) / (2 * len(g))
        cy = sum((boxes[i][1] + boxes[i][3]) for i in g) / (2 * len(g))
        return (cx, cy)

    final_groups.sort(key=group_key2)
    return merge_fragments(final_groups, boxes, med_h)


def merge_fragments(groups, boxes, med_h):
    """把瓶颈/瓶盖等小碎片合并到最近的主瓶。
    碎片判定：行数 <= 2 且 bbox 面积 < 主瓶中位数面积的 30%。
    合并目标：在 x 方向重叠或接近、y 方向最近的非碎片组。"""
    if len(groups) <= 1:
        return groups

    def bbox(g):
        x1 = min(boxes[i][0] for i in g)
        y1 = min(boxes[i][1] for i in g)
        x2 = max(boxes[i][2] for i in g)
        y2 = max(boxes[i][3] for i in g)
        return x1, y1, x2, y2

    areas = [(x2 - x1) * (y2 - y1) for g in groups for x1, y1, x2, y2 in [bbox(g)]]
    if not areas:
        return groups
    sorted_areas = sorted(areas)
    median_area = sorted_areas[len(sorted_areas) // 2]
    frag_area_thresh = max(median_area * 0.3, med_h * med_h * 4)

    is_frag = [len(g) <= 3 and areas[i] < frag_area_thresh for i, g in enumerate(groups)]
    if not any(is_frag):
        return groups

    # 每个碎片找最近的非碎片：x 投影重叠优先，其次 y 距离
    merged = list(groups)
    for i, g in enumerate(groups):
        if not is_frag[i]:
            continue
        fx1, fy1, fx2, fy2 = bbox(g)
        fw = fx2 - fx1
        fcx = (fx1 + fx2) / 2
        fcy = (fy1 + fy2) / 2
        best_j = -1
        best_score = float("inf")
        for j, g2 in enumerate(groups):
            if j == i or is_frag[j]:
                continue
            bx1, by1, bx2, by2 = bbox(g2)
            bcx = (bx1 + bx2) / 2
            bcy = (by1 + by2) / 2
            x_overlap = max(0, min(fx2, bx2) - max(fx1, bx1))
            overlap_ratio = x_overlap / max(fw, 1)
            x_dist = abs(fcx - bcx)
            y_dist = abs(fcy - bcy)
            # x 重叠过半时强烈倾向合并（瓶颈/瓶盖文字）
            if overlap_ratio > 0.5:
                score = y_dist * 0.1
            else:
                score = y_dist + x_dist * 0.5 - x_overlap * 2
            if score < best_score:
                best_score = score
                best_j = j
        if best_j >= 0 and best_score < med_h * 30:
            merged[best_j] = merged[best_j] + g
            merged[i] = None

    merged = [g for g in merged if g is not None]
    # 再合并一次：处理碎片互相合并后产生的新小簇
    if len(merged) < len(groups):
        boxes2 = boxes
        return merge_fragments(merged, boxes2, med_h)
    # 重新按左->右、上->下排序
    def gkey(g):
        cx = sum((boxes[i][0] + boxes[i][2]) for i in g) / (2 * len(g))
        cy = sum((boxes[i][1] + boxes[i][3]) for i in g) / (2 * len(g))
        return (cx, cy)
    merged.sort(key=gkey)
    return merged


def build_bottle(no, idxs, lines):
    group = [lines[i] for i in idxs]
    group.sort(key=lambda ln: (ln["box"][1], ln["box"][0]))
    x1 = min(ln["box"][0] for ln in group)
    y1 = min(ln["box"][1] for ln in group)
    x2 = max(ln["box"][2] for ln in group)
    y2 = max(ln["box"][3] for ln in group)

    raw = " ".join(ln["text"] for ln in group)

    # 候选酒名：滤掉度数/容量/厂家信息等噪声行，取信息量大的前两行
    clean = [ln["text"] for ln in group if not NOISE_RE.search(ln["text"])]
    clean = [re.sub(r"\s+", " ", t).strip() for t in clean if t.strip()]
    name_hint = " / ".join(sorted(clean, key=len, reverse=True)[:2]) if clean else ""

    return {
        "bottle": no,
        "bbox": [x1, y1, x2, y2],
        "name_hint": name_hint,
        "raw_text": raw,
        "lines": [{"text": ln["text"], "score": ln["score"]} for ln in group],
    }


def load_aliases(html_path):
    """从 index.html 提取 ALIASES 与 id->中文名/分类，返回 dict 或 None。"""
    node = shutil.which("node") or "/Users/bytedance/.local/node22/bin/node"
    if not os.path.exists(node):
        return None
    script = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
function grab(varName) {
  const start = html.indexOf('var ' + varName);
  if (start < 0) throw new Error('not found: ' + varName);
  const eq = html.indexOf('=', start);
  const open = html.indexOf('{', eq);
  const openArr = html.indexOf('[', eq);
  let begin, opener, closer;
  if (openArr >= 0 && (open < 0 || openArr < open)) { begin = openArr; opener = '['; closer = ']'; }
  else { begin = open; opener = '{'; closer = '}'; }
  let depth = 0;
  for (let i = begin; i < html.length; i++) {
    const ch = html[i];
    if (ch === opener) depth++;
    else if (ch === closer) { depth--; if (depth === 0) return html.slice(begin, i + 1); }
  }
  throw new Error('unbalanced: ' + varName);
}
const CATS = (new Function('return ' + grab('CATS')))();
const ALIASES = (new Function('return ' + grab('ALIASES')))();
const id2name = {}, id2cat = {};
for (const c of CATS) for (const it of c.items) { id2name[it[0]] = it[1]; id2cat[it[0]] = c.name; }
process.stdout.write(JSON.stringify({ aliases: ALIASES, id2name, id2cat }));
"""
    try:
        out = subprocess.run([node, "-e", script, html_path],
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            print("[warn] alias extract failed: " + out.stderr[:200], file=sys.stderr)
            return None
        return json.loads(out.stdout)
    except Exception as e:
        print("[warn] alias load skipped: %s" % e, file=sys.stderr)
        return None


def match_brands(bottle, alias_db):
    """复刻网站匹配规则：拉丁词边界 + 中文去空格子串。每瓶独立匹配。"""
    if not alias_db:
        return []
    text = bottle["raw_text"].lower()
    text_nospace = re.sub(r"\s+", "", text)
    hits = []
    for rid, aliases in alias_db["aliases"].items():
        for alias in aliases:
            a = alias.lower().strip()
            if not a:
                continue
            if re.search(r"[a-z0-9]", a):
                pat = re.escape(a).replace(r"\ ", r"\s+")
                if re.search(r"(?<![a-z0-9])" + pat + r"(?![a-z0-9])", text):
                    hits.append(rid)
                    break
            else:
                if re.sub(r"\s+", "", a) in text_nospace:
                    hits.append(rid)
                    break
    seen = set()
    result = []
    for rid in hits:
        if rid in seen:
            continue
        seen.add(rid)
        result.append({
            "id": rid,
            "name": alias_db["id2name"].get(rid, rid),
            "category": alias_db["id2cat"].get(rid, ""),
        })
    return result


def draw_vis(image_path, bottles, out_path):
    import cv2
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    pad = max(8, int(w * 0.01))
    for b in bottles:
        x1, y1, x2, y2 = b["bbox"]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 200, 90), 3)
        label = "#%d" % b["bottle"]
        if b.get("matches"):
            label += " " + ",".join(m["id"] for m in b["matches"][:3])
        cv2.putText(img, label, (x1, max(28, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 200, 90), 2)
    cv2.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser(description="多酒瓶照片逐瓶 OCR + 品牌匹配")
    ap.add_argument("image", help="酒瓶照片路径")
    ap.add_argument("--html", help="网站 index.html 路径（用于品牌别名匹配）")
    ap.add_argument("--vis", help="输出画框可视化图路径")
    ap.add_argument("--pretty", action="store_true", help="人类可读输出")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    html_path = args.html or os.path.join(here, "..", "index.html")

    import cv2
    img = cv2.imread(args.image)
    if img is None:
        print("无法读取图片: %s" % args.image, file=sys.stderr)
        sys.exit(1)
    img_h, img_w = img.shape[:2]

    ocr = load_ocr()
    lines = run_ocr(ocr, args.image)

    if not lines:
        print(json.dumps({"image": args.image, "bottle_count": 0, "bottles": []},
                         ensure_ascii=False))
        return

    groups = cluster_bottles(lines, img_w, img_h)
    bottles = [build_bottle(i + 1, g, lines) for i, g in enumerate(groups)]

    alias_db = load_aliases(html_path) if os.path.exists(html_path) else None
    for b in bottles:
        b["matches"] = match_brands(b, alias_db)
        b["matched_label"] = "、".join(m["name"] for m in b["matches"]) if b["matches"] else ""

    out = {
        "image": args.image,
        "image_size": {"width": img_w, "height": img_h},
        "bottle_count": len(bottles),
        "bottles": bottles,
    }

    if args.pretty:
        print("图片：%s（%dx%d）" % (args.image, img_w, img_h))
        print("共区分出 %d 个酒瓶" % len(bottles))
        print("=" * 60)
        for b in bottles:
            print("【酒瓶 #%d】 区域%s" % (b["bottle"], b["bbox"]))
            print("  候选酒名：%s" % (b["name_hint"] or "（未提取到）"))
            if b["matches"]:
                print("  匹配库存：%s" % "、".join(
                    "%s[%s]" % (m["name"], m["category"]) for m in b["matches"]))
            else:
                print("  匹配库存：无（清单外新酒，建议按候选名自定义添加）")
            print("  识别原文：")
            for ln in b["lines"]:
                print("    · %-30s  (%.2f)" % (ln["text"], ln["score"]))
            print("-" * 60)
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.vis:
        draw_vis(args.image, bottles, args.vis)
        print("[vis] 已保存分瓶标注图：%s" % args.vis, file=sys.stderr)


if __name__ == "__main__":
    main()
