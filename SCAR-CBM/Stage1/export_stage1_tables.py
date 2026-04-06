#!/usr/bin/env python3
"""
导出 Stage 1 的三个结构化表格：
1) 概念共现关系表 (Concept Co-occurrence Table)
2) 超边构成表 (Hyperedge Composition Table)
3) 概念节点统计表 (Concept Statistics Table)

示例：
  cd SCAR-CBM
  python Stage1/export_stage1_tables.py \
      --from-dir Stage1/stage1_cub_output \
      --out Stage1/stage1_cub_output/tables
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Iterable

import numpy as np
import torch


def read_meta(path: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not os.path.isfile(path):
        return meta
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def _clean_concept_name(raw: str) -> str:
    # 便于阅读：去掉属性 id 前缀，并将下划线替换为空格
    cleaned = re.sub(r"^\d+\s+", "", raw.strip())
    return cleaned.replace("_", " ")


def load_concept_names(
    num_concepts: int, cub_root: str | None
) -> tuple[list[str], str | None]:
    names = [f"concept_{i}" for i in range(num_concepts)]
    if not cub_root:
        return names, None
    attr_txt = os.path.join(cub_root, "attributes", "attributes.txt")
    if not os.path.isfile(attr_txt):
        return names, None

    loaded: list[str] = []
    with open(attr_txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            loaded.append(_clean_concept_name(line))
    if len(loaded) >= num_concepts:
        return loaded[:num_concepts], attr_txt
    for i, name in enumerate(loaded):
        names[i] = name
    return names, attr_txt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(path: str, header: list[str], rows: Iterable[list[object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def load_adj_matrix(path: str) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "detach"):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)


def build_cooccurrence_table(
    cooc: np.ndarray,
    adj: np.ndarray,
    concept_names: list[str],
) -> list[list[object]]:
    n = cooc.shape[0]
    rows: list[list[object]] = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(cooc[i, j])
            w = float(adj[i, j])
            if c <= 0 and w <= 0:
                continue
            rows.append(
                [
                    i,
                    concept_names[i],
                    j,
                    concept_names[j],
                    c,
                    w,
                ]
            )
    rows.sort(key=lambda r: (r[4], r[5]), reverse=True)
    return rows


def build_hyperedge_table_from_json(
    hypergraph_json: dict,
    concept_names: list[str],
) -> tuple[list[list[object]], np.ndarray]:
    num_concepts = len(concept_names)
    incident = np.zeros(num_concepts, dtype=np.int64)
    rows: list[list[object]] = []
    edges = hypergraph_json.get("edges", [])
    for idx, e in enumerate(edges):
        u = int(e["u"])
        v = int(e["v"])
        w = float(e.get("weight", 1.0))
        incident[u] += 1
        incident[v] += 1
        rows.append(
            [
                idx,
                f"{u}|{v}",
                f"{concept_names[u]} | {concept_names[v]}",
                2,
                w,
            ]
        )
    rows.sort(key=lambda r: r[4], reverse=True)
    return rows, incident


def build_concept_statistics_table(
    concept_names: list[str],
    incident_counts: np.ndarray,
) -> list[list[object]]:
    rows: list[list[object]] = []
    for i, name in enumerate(concept_names):
        cnt = int(incident_counts[i])
        rows.append(
            [
                i,
                name,
                cnt,
                "是" if cnt == 0 else "否",
            ]
        )
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Stage 1 三张表格")
    parser.add_argument(
        "--from-dir",
        type=str,
        default="Stage1/stage1_cub_output",
        help="Stage 1 输出目录（需包含 c_labeled_train.npy / hypergraph_edges.json）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="表格输出目录，默认 <from-dir>/tables",
    )
    parser.add_argument(
        "--cub-root",
        type=str,
        default=None,
        help="CUB 根目录（用于读取 attributes.txt 概念名）；不传则尝试从 meta 自动读取",
    )
    args = parser.parse_args()

    from_dir = os.path.abspath(args.from_dir)
    out_dir = os.path.abspath(args.out or os.path.join(from_dir, "tables"))
    ensure_dir(out_dir)

    c_path = os.path.join(from_dir, "c_labeled_train.npy")
    cooc_path = os.path.join(from_dir, "cooccurrence_raw.npy")
    adj_path = os.path.join(from_dir, "adj_norm_gcn.pt")
    edge_json_path = os.path.join(from_dir, "hypergraph_edges.json")
    meta_path = os.path.join(from_dir, "cub_stage1_meta.txt")

    if not os.path.isfile(c_path):
        raise FileNotFoundError(f"缺少文件: {c_path}")
    if not os.path.isfile(edge_json_path):
        raise FileNotFoundError(f"缺少文件: {edge_json_path}")
    if not os.path.isfile(adj_path):
        raise FileNotFoundError(f"缺少文件: {adj_path}")

    concept_ids = np.load(c_path).astype(np.float32)
    num_concepts = int(concept_ids.shape[1])

    if os.path.isfile(cooc_path):
        cooc = np.load(cooc_path).astype(np.float64)
    else:
        cooc = (concept_ids.T @ concept_ids).astype(np.float64)
    adj = load_adj_matrix(adj_path).astype(np.float64)

    meta = read_meta(meta_path)
    cub_root = args.cub_root or meta.get("cub_root")
    concept_names, concept_name_source = load_concept_names(num_concepts, cub_root)

    with open(edge_json_path, "r", encoding="utf-8") as f:
        hypergraph_json = json.load(f)

    cooc_rows = build_cooccurrence_table(cooc, adj, concept_names)
    edge_rows, incident_counts = build_hyperedge_table_from_json(
        hypergraph_json, concept_names
    )
    stat_rows = build_concept_statistics_table(concept_names, incident_counts)

    cooc_csv = os.path.join(out_dir, "concept_cooccurrence_table.csv")
    edge_csv = os.path.join(out_dir, "hyperedge_composition_table.csv")
    stat_csv = os.path.join(out_dir, "concept_statistics_table.csv")

    save_csv(
        cooc_csv,
        [
            "concept_a_id",
            "concept_a",
            "concept_b_id",
            "concept_b",
            "cooccurrence_count",
            "gcn_adj_weight",
        ],
        cooc_rows,
    )
    save_csv(
        edge_csv,
        [
            "hyperedge_id",
            "concept_ids",
            "concepts",
            "degree",
            "hyperedge_weight",
        ],
        edge_rows,
    )
    save_csv(
        stat_csv,
        [
            "concept_id",
            "concept_name",
            "participating_hyperedges",
            "isolated",
        ],
        stat_rows,
    )

    print("Stage 1 表格导出完成：")
    print(f"  - {cooc_csv} ({len(cooc_rows)} rows)")
    print(f"  - {edge_csv} ({len(edge_rows)} rows)")
    print(f"  - {stat_csv} ({len(stat_rows)} rows)")
    if concept_name_source:
        print(f"  - 使用概念名来源: {concept_name_source}")
    else:
        print("  - 未加载到 CUB attributes.txt，概念名使用 concept_<id> 占位符")


if __name__ == "__main__":
    main()
