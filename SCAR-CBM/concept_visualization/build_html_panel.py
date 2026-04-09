#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description='根据 manifest.json 生成 concept 可视化 HTML 面板')
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    out_path = os.path.abspath(args.out)
    panel_dir = os.path.dirname(out_path)
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data['meta']
    summary = data['summary']
    samples = data['samples']

    def img(relpath: str, alt: str) -> str:
        return f'<img src="{html.escape(relpath)}" alt="{html.escape(alt)}" loading="lazy">'

    sample_blocks = []
    for sample in samples:
        concept_text = ', '.join(html.escape(x) for x in sample['concept_labels'])
        sample_blocks.append(
            f'''
            <section class="sample-card">
              <h3>row {sample['row_index']} | {html.escape(sample['relpath'])}</h3>
              <p class="concepts">{concept_text}</p>
              <div class="grid three">
                <figure>{img(sample['stage2_overlay'], 'stage2 overlay')}<figcaption>Stage2 Overlay</figcaption></figure>
                <figure>{img(sample['stage5_overlay'], 'stage5 overlay')}<figcaption>Stage5 Overlay</figcaption></figure>
                <figure>{img(sample['compare_panel'], 'compare panel')}<figcaption>Stage2 vs Stage5 vs Delta</figcaption></figure>
              </div>
            </section>
            '''
        )

    html_text = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SCAR-CBM Concept Visualization</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1c1c1c;
      --muted: #645f57;
      --line: #d8cfbf;
      --accent: #0d6c66;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(13,108,102,0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(193,124,49,0.14), transparent 30%),
        var(--bg);
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 24px 80px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    p {{ margin: 0 0 12px; line-height: 1.5; }}
    .hero, .section, .sample-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(28,28,28,0.06);
    }}
    .hero {{ padding: 24px; }}
    .section {{ margin-top: 18px; padding: 20px; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px 18px;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
    }}
    .grid.two {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .grid.three {{ grid-template-columns: 1fr; }}
    .sample-card {{ margin-top: 18px; padding: 18px; }}
    .concepts {{
      color: var(--accent);
      font-weight: 600;
      margin-bottom: 14px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: #fff;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    figcaption {{
      padding: 10px 12px;
      font-size: 14px;
      color: var(--muted);
      border-top: 1px solid var(--line);
    }}
    ol {{
      margin: 0;
      padding-left: 20px;
    }}
    li {{ margin: 0 0 8px; }}
    @media (min-width: 980px) {{
      .grid.three {{ grid-template-columns: repeat(3, 1fr); }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>SCAR-CBM Concept Visualization Panel</h1>
      <p>这个面板展示了 Stage2 空间概念热图、Stage5 最终权重热图，以及它们在同一批样本上的差异。</p>
      <div class="meta">
        <div><strong>Stage2 NPZ</strong><br>{html.escape(meta['stage2_npz'])}</div>
        <div><strong>Stage5 Weights</strong><br>{html.escape(meta['stage5_weights'])}</div>
        <div><strong>Sample Count</strong><br>{meta['sample_count']}</div>
        <div><strong>Top-K Concepts</strong><br>{meta['top_k']}</div>
      </div>
    </section>

    <section class="section">
      <h2>Summary</h2>
      <div class="grid two">
        <figure>{img(summary['top_stage2_chart'], 'top stage2 concepts')}<figcaption>Stage2 Top Concepts</figcaption></figure>
        <figure>{img(summary['top_stage5_chart'], 'top stage5 concepts')}<figcaption>Stage5 Top Concepts</figcaption></figure>
      </div>
      <div class="grid two" style="margin-top:16px">
        <div>
          <h3>Stage2 Frequent Concepts</h3>
          <ol>
            {''.join(f"<li>{html.escape(row['concept_name'])} ({row['count']})</li>" for row in summary['top_stage2'])}
          </ol>
        </div>
        <div>
          <h3>Stage5 Frequent Concepts</h3>
          <ol>
            {''.join(f"<li>{html.escape(row['concept_name'])} ({row['count']})</li>" for row in summary['top_stage5'])}
          </ol>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Samples</h2>
      {''.join(sample_blocks)}
    </section>
  </main>
</body>
</html>
'''

    os.makedirs(panel_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print(f'已写入: {out_path}')


if __name__ == '__main__':
    main()
