#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch

from compare_spatial_heatmaps import build_models, load_adj, run_one
from visualize_spatial_concept_heatmap import load_image_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description='基于 manifest 对同一批样本生成 with-mask vs no-mask 对比面板')
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--before-weights', type=str, required=True)
    parser.add_argument('--after-weights', type=str, required=True)
    parser.add_argument('--before-label', type=str, default='with_mask')
    parser.add_argument('--after-label', type=str, default='no_mask')
    parser.add_argument('--stage1-out', type=str, required=True)
    parser.add_argument('--num-concepts', type=int, default=312)
    parser.add_argument('--feature-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--concepts', type=str, default=None)
    parser.add_argument('--top-k', type=int, default=6)
    parser.add_argument('--selection', type=str, choices=('joint', 'delta'), default='delta')
    parser.add_argument('--cmap', type=str, default='turbo')
    parser.add_argument('--delta-cmap', type=str, default='coolwarm')
    parser.add_argument('--alpha', type=float, default=0.45)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    device = torch.device(args.device)
    before_model, after_model = build_models(args, device)
    adj = load_adj(device, args.stage1_out, None)

    root_dir = os.path.dirname(os.path.abspath(args.manifest))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    html_cards = []
    rows = []
    for sample in manifest['samples']:
        relpath = sample['relpath']
        image_path = os.path.join(manifest['meta']['cub_root'], 'images', relpath)
        x = load_image_tensor(image_path, args.image_size).to(device)
        stem = f"row_{sample['row_index']:04d}"
        out_png = os.path.join(out_dir, f'{stem}_mask_vs_nomask.png')
        run_one(x, relpath, out_png, before_model, after_model, adj, args)
        out_json = os.path.splitext(out_png)[0] + '.json'
        rows.append({'row_index': sample['row_index'], 'relpath': relpath, 'png': os.path.basename(out_png), 'json': os.path.basename(out_json)})
        html_cards.append(
            f'''<section class="card"><h3>row {sample['row_index']} | {relpath}</h3><img src="{os.path.basename(out_png)}" alt="{relpath}"></section>'''
        )

    summary_path = os.path.join(out_dir, 'compare_manifest.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'manifest': os.path.abspath(args.manifest), 'rows': rows}, f, ensure_ascii=False, indent=2)

    html = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mask vs No-Mask Compare</title>
  <style>
    body {{ margin: 0; background: #f4efe6; color: #1f1f1f; font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif; }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 28px; }}
    .hero, .card {{ background: #fffdf8; border: 1px solid #d8cfbf; border-radius: 18px; box-shadow: 0 10px 26px rgba(0,0,0,0.05); }}
    .hero {{ padding: 22px; margin-bottom: 18px; }}
    .grid {{ display: grid; gap: 18px; }}
    img {{ display: block; width: 100%; height: auto; border-radius: 14px; }}
    .card {{ padding: 16px; }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    p {{ margin: 0; line-height: 1.5; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>With-Mask vs No-Mask</h1>
      <p>同一批样本，对比带 segmentation 训练权重与 contour_mask=None 训练权重的空间 concept 热图差异。</p>
      <p>before: {args.before_weights}</p>
      <p>after: {args.after_weights}</p>
    </section>
    <section class="grid">
      {''.join(html_cards)}
    </section>
  </main>
</body>
</html>
'''
    html_path = os.path.join(out_dir, 'index.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'已写入: {summary_path}')
    print(f'已写入: {html_path}')


if __name__ == '__main__':
    main()
