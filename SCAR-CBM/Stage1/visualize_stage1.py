"""
Stage 1 可视化工具：概念关系图的构建和可视化。

从 run_cub_stage1 输出目录可视化（示例）:
  cd SCAR-CBM
  python Stage1/visualize_stage1.py --from-dir Stage1/stage1_cub_output --out Stage1/stage1_cub_output/viz --no-show
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import networkx as nx
import torch
from matplotlib.ticker import MaxNLocator

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core_modules import ConceptHypergraph

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def configure_chinese_font(font_path: str | None = None) -> None:
    """
    配置中文字体，避免图注乱码/缺字。

    优先级：
    1) --font-path 指定字体文件
    2) 系统已安装常见中文字体
    3) 若都不可用，给出提示（图仍可保存，但中文可能缺字）
    """
    candidates = [
        # Noto CJK (Debian/Ubuntu fonts-noto-cjk 常见命名)
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK HK",
        "Noto Sans CJK JP",
        "Noto Sans CJK KR",
        "Noto Serif CJK SC",
        "Noto Serif CJK TC",
        "Noto Serif CJK HK",
        "Noto Serif CJK JP",
        "Noto Serif CJK KR",
        "Noto Sans Mono CJK SC",
        "Noto Sans Mono CJK TC",
        "Noto Sans Mono CJK HK",
        "Noto Sans Mono CJK JP",
        "Noto Sans Mono CJK KR",
        # 其他常见中文字体族名
        "Source Han Sans SC",
        "Source Han Serif SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "STHeiti",
    ]
    selected = None

    if font_path:
        if not os.path.isfile(font_path):
            raise FileNotFoundError(f"--font-path 不存在: {font_path}")
        fm.fontManager.addfont(font_path)
        selected = fm.FontProperties(fname=font_path).get_name()
    else:
        def pick_from_installed() -> str | None:
            installed = {f.name for f in fm.fontManager.ttflist}
            for name in candidates:
                if name in installed:
                    return name
            return None

        selected = pick_from_installed()
        if selected is None:
            # 可能是 matplotlib 字体缓存未刷新（如刚 apt 安装字体）
            try:
                fm._rebuild()
            except Exception:
                pass
            selected = pick_from_installed()

    if selected is None:
        print(
            "⚠ 未检测到中文字体；图可能出现中文缺字。"
            "可安装 `fonts-noto-cjk`，或运行时传 --font-path /path/to/font.ttf"
        )
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
        print(f"使用中文字体: {selected}")
    plt.rcParams["axes.unicode_minus"] = False


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_threshold_from_cub_meta(meta_path: str, default: float = 0.01) -> float:
    if not os.path.isfile(meta_path):
        return default
    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "cooccurrence_threshold" in line:
                m = re.search(r"[\d.]+", line.split(":")[-1])
                if m:
                    return float(m.group())
    return default


def load_stage1_output_dir(dir_path: str) -> tuple[np.ndarray, float]:
    """读取 run_cub_stage1.py / run_stage1_train_only 等产生的目录。"""
    dir_path = os.path.abspath(dir_path)
    npy = os.path.join(dir_path, "c_labeled_train.npy")
    if not os.path.isfile(npy):
        raise FileNotFoundError(
            f"目录中缺少 c_labeled_train.npy: {dir_path}\n"
            "请先运行 Stage1/run_cub_stage1.py 或使用 --npy 指定概念矩阵。"
        )
    concept_ids = np.load(npy).astype(np.float32)
    meta = os.path.join(dir_path, "cub_stage1_meta.txt")
    threshold = load_threshold_from_cub_meta(meta)
    return concept_ids, threshold


class Stage1Visualizer:
    """Stage 1 概念关系图可视化"""
    
    def __init__(self, num_concepts: int = 10):
        self.num_concepts = num_concepts
        self.hypergraph = None
        self.cooccurrence_matrix = None
        self.adj_matrix = None
        
    def generate_sample_data(self, num_samples: int = 100):
        """生成示例数据"""
        # 生成概念共现矩阵
        # 模拟：某些概念经常一起出现
        concept_ids = np.zeros((num_samples, self.num_concepts), dtype=np.float32)
        
        # 设置一些概念聚类（会一起出现）
        clusters = {
            'cluster1': [0, 1, 2],      # 动物相关
            'cluster2': [3, 4, 5],      # 颜色相关
            'cluster3': [6, 7, 8],      # 形状相关
            'cluster4': [9],            # 其他
        }
        
        # 为每个样本分配概念
        for i in range(num_samples):
            # 选择一个主聚类
            main_cluster = np.random.choice(list(clusters.keys()))
            concepts = clusters[main_cluster]
            
            # 在聚类内分配概念
            for concept_id in concepts:
                if np.random.rand() > 0.3:  # 80%概率出现
                    concept_ids[i, concept_id] = np.random.rand()
        
        return concept_ids
    
    def build_hypergraph(self, concept_ids: np.ndarray, threshold: float = 0.1):
        """构建概念超图"""
        self.hypergraph = ConceptHypergraph(self.num_concepts)
        self.hypergraph.build_from_labeled_data(
            None, torch.FloatTensor(concept_ids), threshold
        )
        self.cooccurrence_matrix = _to_numpy(self.hypergraph.cooccurrence_matrix)
        self.adj_matrix = _to_numpy(self.hypergraph.get_adjacency_matrix("cpu"))
        
        return self.hypergraph, self.cooccurrence_matrix, self.adj_matrix
    
    def plot_cooccurrence_matrix(self, save_path: str = None):
        """可视化共现矩阵"""
        nc = self.num_concepts
        figsize = (12, 10) if nc > 40 else (10, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        # 归一化用于显示
        cooc_norm = self.cooccurrence_matrix / (self.cooccurrence_matrix.max() + 1e-8)
        
        # 绘制热力图
        im = ax.imshow(cooc_norm, cmap='YlOrRd', aspect='auto')
        
        # 设置标签
        ax.set_xlabel('概念ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('概念ID', fontsize=12, fontweight='bold')
        ax.set_title('Stage 1: 概念共现矩阵\n(颜色深度表示共现频率)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        show_all_ticks = nc <= 32
        if show_all_ticks:
            ax.set_xticks(range(nc))
            ax.set_yticks(range(nc))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(12, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(12, integer=True))
        
        if nc <= 24:
            for i in range(nc):
                for j in range(nc):
                    ax.text(
                        j, i, f'{cooc_norm[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=7,
                    )
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('共现频率', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 已保存: {save_path}")
        
        return fig
    
    def plot_hypergraph_network(self, save_path: str = None, layout: str = 'spring'):
        """可视化概念超图网络"""
        fig, ax = plt.subplots(figsize=(14, 12))
        
        if self.hypergraph is not None and self.hypergraph.hypergraph.number_of_nodes() > 0:
            G = self.hypergraph.hypergraph.copy()
        else:
            G = nx.Graph()
            for i in range(self.num_concepts):
                G.add_node(i)
            for i in range(self.num_concepts):
                for j in range(i + 1, self.num_concepts):
                    weight = self.adj_matrix[i, j]
                    if weight > 0.01:
                        G.add_edge(i, j, weight=weight)
        
        n = G.number_of_nodes()
        if layout == 'spring':
            k_scale = 1.0 / max(np.sqrt(n), 1)
            pos = nx.spring_layout(G, k=k_scale, iterations=35, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # 绘制背景
        ax.set_facecolor('#f8f9fa')
        
        edges = list(G.edges())

        def _edge_w(data):
            w = data.get("weight", 1.0)
            if hasattr(w, "item"):
                return float(w.item())
            return float(w)

        weights = [_edge_w(G[u][v]) for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [max(0.3, 3 * w / max_weight) for w in weights]
        edge_colors = [w / max_weight for w in weights]
        
        edges_collection = nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color=edge_colors,
            edge_cmap=plt.cm.Blues,
            edge_vmin=0,
            edge_vmax=1,
            ax=ax,
            alpha=0.6
        )
        
        node_list = list(G.nodes())
        node_colors = [G.degree(nidx) for nidx in node_list]
        node_size = 800 if n <= 40 else max(40, 2400 // max(n, 1))
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=node_list,
            node_color=node_colors,
            node_size=node_size,
            cmap='RdYlGn',
            ax=ax,
            alpha=0.9,
            edgecolors='black',
            linewidths=0.5 if n > 60 else 2,
        )
        
        if n <= 45:
            labels = {i: f'{i}' for i in node_list}
            nx.draw_networkx_labels(
                G, pos,
                labels,
                font_size=8 if n > 20 else 10,
                font_weight='bold',
                font_color='white',
                ax=ax,
            )
        
        ax.set_title(
            f'Stage 1: 概念超图网络 (节点数={n}, 边数={G.number_of_edges()})\n'
            '(节点颜色表示连接度)',
            fontsize=14,
            fontweight='bold',
            pad=20,
        )
        ax.axis('off')
        
        legend_text = (
            '节点: 概念 ID\n'
            '边: 共现超图边权重\n'
            '颜色: 连接度\n'
            + ('概念数较多时已隐藏节点文字。' if n > 45 else '')
        )
        ax.text(0.02, 0.98, legend_text, 
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 已保存: {save_path}")
        
        return fig, G
    
    def plot_adjacency_matrix(self, save_path: str = None):
        """可视化邻接矩阵（GCN输入）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 第一个子图：原始邻接矩阵
        im1 = axes[0].imshow(self.adj_matrix, cmap='Blues', aspect='auto')
        axes[0].set_xlabel('概念ID', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('概念ID', fontsize=11, fontweight='bold')
        axes[0].set_title('邻接矩阵（度数归一化）', fontsize=12, fontweight='bold')
        if self.num_concepts <= 40:
            axes[0].set_xticks(range(self.num_concepts))
            axes[0].set_yticks(range(self.num_concepts))
        else:
            axes[0].xaxis.set_major_locator(MaxNLocator(12, integer=True))
            axes[0].yaxis.set_major_locator(MaxNLocator(12, integer=True))
        plt.colorbar(im1, ax=axes[0], label='权重')
        
        # 第二个子图：二值化邻接矩阵（只显示存在/不存在）
        adj_binary = (self.adj_matrix > 0.01).astype(int)
        im2 = axes[1].imshow(adj_binary, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1].set_xlabel('概念ID', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('概念ID', fontsize=11, fontweight='bold')
        axes[1].set_title('二值邻接矩阵（0/1）', fontsize=12, fontweight='bold')
        if self.num_concepts <= 40:
            axes[1].set_xticks(range(self.num_concepts))
            axes[1].set_yticks(range(self.num_concepts))
        else:
            axes[1].xaxis.set_major_locator(MaxNLocator(12, integer=True))
            axes[1].yaxis.set_major_locator(MaxNLocator(12, integer=True))
        plt.colorbar(im2, ax=axes[1], label='连接', ticks=[0, 1])
        
        fig.suptitle('Stage 1: 邻接矩阵（GCN的输入）', 
                     fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 已保存: {save_path}")
        
        return fig
    
    def plot_concept_statistics(self, concept_ids: np.ndarray, save_path: str = None):
        """可视化概念统计信息"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        nc = self.num_concepts
        
        # 1. 每个概念出现的频率
        concept_freq = concept_ids.mean(axis=0)
        ax = axes[0, 0]
        if nc <= 48:
            bars = ax.bar(range(nc), concept_freq, color='steelblue', alpha=0.7)
            ax.set_xticks(range(nc))
            if nc <= 20:
                for bar, freq in zip(bars, concept_freq):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f'{freq:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                    )
        else:
            ax.plot(np.arange(nc), concept_freq, color='steelblue', linewidth=1.2)
            ax.fill_between(np.arange(nc), concept_freq, alpha=0.25, color='steelblue')
            ax.xaxis.set_major_locator(MaxNLocator(12, integer=True))
        ax.set_xlabel('概念ID', fontsize=11, fontweight='bold')
        ax.set_ylabel('出现频率', fontsize=11, fontweight='bold')
        ax.set_title('概念出现频率', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 2. 每个样本的概念数
        concepts_per_sample = (concept_ids > 0).sum(axis=1)
        ax = axes[0, 1]
        ax.hist(concepts_per_sample, bins=min(40, int(concepts_per_sample.max()) + 1), color='coral', alpha=0.7, edgecolor='black')
        ax.set_xlabel('样本包含的概念数', fontsize=11, fontweight='bold')
        ax.set_ylabel('样本数', fontsize=11, fontweight='bold')
        ax.set_title('每个样本的概念数分布', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. 概念关系强度分布
        ax = axes[1, 0]
        # 只取上三角矩阵的非零元素
        adj_values = self.adj_matrix[np.triu_indices_from(self.adj_matrix, k=1)]
        adj_values = adj_values[adj_values > 0]
        ax.hist(adj_values, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        ax.set_xlabel('关系权重', fontsize=11, fontweight='bold')
        ax.set_ylabel('频数', fontsize=11, fontweight='bold')
        ax.set_title('概念关系权重分布', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. 概念连接度分布
        ax = axes[1, 1]
        node_degrees = (self.adj_matrix > 0.01).sum(axis=1)
        if nc <= 48:
            bars = ax.bar(range(nc), node_degrees, color='mediumpurple', alpha=0.7)
            ax.set_xticks(range(nc))
            if nc <= 20:
                for bar, degree in zip(bars, node_degrees):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f'{int(degree)}',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                    )
        else:
            ax.plot(np.arange(nc), node_degrees, color='mediumpurple', linewidth=1.2)
            ax.xaxis.set_major_locator(MaxNLocator(12, integer=True))
        ax.set_xlabel('概念ID', fontsize=11, fontweight='bold')
        ax.set_ylabel('连接度', fontsize=11, fontweight='bold')
        ax.set_title('概念连接度（度数）', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        fig.suptitle('Stage 1: 概念统计分析', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 已保存: {save_path}")
        
        return fig
    
    def plot_complete_pipeline(self, concept_ids: np.ndarray, save_path: str = None):
        """绘制完整的Stage 1流程图"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # 标题
        fig.suptitle('Stage 1: 概念关系图构建完整流程', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. 输入数据
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.7, '输入数据', ha='center', va='center', 
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.7))
        ax1.text(0.5, 0.3, f'有标签数据\n{concept_ids.shape[0]} 个样本\n{concept_ids.shape[1]} 个概念',
                ha='center', va='center', fontsize=10)
        ax1.axis('off')
        
        # 箭头
        ax_arrow1 = fig.add_subplot(gs[0, 1])
        ax_arrow1.text(0.5, 0.5, '→', ha='center', va='center', fontsize=30)
        ax_arrow1.axis('off')
        
        # 2. 共现矩阵
        ax2 = fig.add_subplot(gs[0, 2])
        cooc_norm = self.cooccurrence_matrix / (self.cooccurrence_matrix.max() + 1e-8)
        im = ax2.imshow(cooc_norm, cmap='YlOrRd', aspect='auto')
        ax2.set_title('共现矩阵', fontsize=12, fontweight='bold')
        ax2.set_xlabel('概念j', fontsize=9)
        ax2.set_ylabel('概念i', fontsize=9)
        ax2.set_xticks([0, 4, 9])
        ax2.set_yticks([0, 4, 9])
        
        # 3. 流程说明
        ax3 = fig.add_subplot(gs[1, :])
        process_text = '''
        处理步骤：
        1️⃣  计算共现矩阵：M[i,j] = Σ(c[i] * c[j])  →  统计哪些概念经常一起出现
        2️⃣  构建超图：根据共现频率，在高频概念间添加边  →  形成概念关系网络
        3️⃣  生成邻接矩阵：对超图的邻接矩阵进行度数归一化  →  用于GCN推理
        4️⃣  输出：可用于Stage 5的GCN进行概念关系推理
        '''
        ax3.text(0.05, 0.5, process_text, fontsize=11, verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax3.axis('off')
        
        # 4. 超图网络（简化版）
        ax4 = fig.add_subplot(gs[2, 0])
        G = nx.Graph()
        for i in range(self.num_concepts):
            G.add_node(i)
        for i in range(self.num_concepts):
            for j in range(i+1, self.num_concepts):
                if self.adj_matrix[i, j] > 0.05:
                    G.add_edge(i, j)
        
        pos = nx.spring_layout(G, k=1, iterations=30, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color='lightgreen', 
                              node_size=300, ax=ax4, alpha=0.9, edgecolors='black')
        nx.draw_networkx_edges(G, pos, ax=ax4, alpha=0.5, width=1.5)
        labels = {i: f'C{i}' for i in range(self.num_concepts)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax4)
        ax4.set_title('超图网络', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # 5. 邻接矩阵
        ax5 = fig.add_subplot(gs[2, 1])
        im5 = ax5.imshow(self.adj_matrix, cmap='Blues', aspect='auto')
        ax5.set_title('邻接矩阵\n(GCN输入)', fontsize=12, fontweight='bold')
        ax5.set_xticks([0, 4, 9])
        ax5.set_yticks([0, 4, 9])
        
        # 6. 输出说明
        ax6 = fig.add_subplot(gs[2, 2])
        output_text = '''
        输出：
        
        ✓ adj_matrix
          [10×10]矩阵
          用于Stage 5的GCN
          
        ✓ hypergraph
          概念关系图对象
          
        使用：
        adj = framework.hypergraph
              .get_adjacency_matrix()
        '''
        ax6.text(0.05, 0.5, output_text, fontsize=9, verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        ax6.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 已保存: {save_path}")
        
        return fig


def export_visualizations(
    visualizer: Stage1Visualizer,
    concept_ids: np.ndarray,
    out_dir: str,
    *,
    show_plots: bool = True,
    with_pipeline: bool = False,
    prefix: str = "",
) -> None:
    """将各类图保存到 out_dir；prefix 用于文件名前缀明细。"""
    os.makedirs(out_dir, exist_ok=True)
    p = lambda name: os.path.join(out_dir, f"{prefix}{name}")

    print("\n[共现矩阵]")
    visualizer.plot_cooccurrence_matrix(save_path=p("stage1_cooccurrence_matrix.png"))
    if show_plots:
        plt.show()
    plt.close("all")

    print("[超图网络]")
    visualizer.plot_hypergraph_network(save_path=p("stage1_hypergraph_network.png"))
    if show_plots:
        plt.show()
    plt.close("all")

    print("[邻接矩阵]")
    visualizer.plot_adjacency_matrix(save_path=p("stage1_adjacency_matrix.png"))
    if show_plots:
        plt.show()
    plt.close("all")

    print("[统计]")
    visualizer.plot_concept_statistics(concept_ids, save_path=p("stage1_statistics.png"))
    if show_plots:
        plt.show()
    plt.close("all")

    if with_pipeline:
        if visualizer.num_concepts > 30:
            print("[跳过流程总览] 概念维度过大，请省略 --with-pipeline 或仅用前文四张图。")
        else:
            print("[流程总览]")
            visualizer.plot_complete_pipeline(
                concept_ids, save_path=p("stage1_complete_pipeline.png")
            )
            if show_plots:
                plt.show()
            plt.close("all")

    print(f"\n✓ 图表已写入: {out_dir}")


def run_demo() -> None:
    """内置 10 概念小示例"""
    print("=" * 60)
    print("Stage 1 可视化演示（随机小示例）")
    print("=" * 60)

    visualizer = Stage1Visualizer(num_concepts=10)
    print("\n生成示例数据…")
    concept_ids = visualizer.generate_sample_data(num_samples=100)
    print(f"数据形状: {concept_ids.shape}")
    print("\n构建概念超图…")
    hg, _, _ = visualizer.build_hypergraph(concept_ids, threshold=0.1)
    print(f"超图边数: {hg.hypergraph.number_of_edges()}")

    cwd = os.getcwd()
    export_visualizations(
        visualizer,
        concept_ids,
        out_dir=cwd,
        show_plots=True,
        with_pipeline=True,
        prefix="",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 可视化；可用 stage1_cub_output 等目录中的 c_labeled_train.npy"
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default=None,
        help="可选：指定中文字体文件路径（.ttf/.otf），用于修复图注中文不显示",
    )
    parser.add_argument(
        "--from-dir",
        type=str,
        default=None,
        help="Stage 1 输出目录（需含 c_labeled_train.npy），如 Stage1/stage1_cub_output",
    )
    parser.add_argument(
        "--npy",
        type=str,
        default=None,
        help="直接指定概念矩阵 .npy [N,C]（与 --from-dir 二选一）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="共现阈值；默认从 cub_stage1_meta.txt 读取，否则 0.1（--npy 且无 meta 时）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="图片保存目录；--from-dir 时默认为 <该目录>/viz",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存 png，不弹出交互窗口（适合服务器）",
    )
    parser.add_argument(
        "--with-pipeline",
        action="store_true",
        help="额外保存完整流程拼图（仅适合概念数较少时）",
    )
    args = parser.parse_args()
    configure_chinese_font(args.font_path)

    if args.from_dir and args.npy:
        raise SystemExit("请只指定 --from-dir 或 --npy 之一")

    if args.from_dir:
        concept_ids, th_meta = load_stage1_output_dir(args.from_dir)
        th = args.threshold if args.threshold is not None else th_meta
        out_dir = args.out or os.path.join(os.path.abspath(args.from_dir), "viz")
        print("=" * 60)
        print("从 Stage 1 输出目录加载")
        print("=" * 60)
        print(f"概念矩阵: {concept_ids.shape}, 阈值: {th}")
        nc = concept_ids.shape[1]
        visualizer = Stage1Visualizer(num_concepts=nc)
        visualizer.build_hypergraph(concept_ids, threshold=th)
        print(f"超图边数: {visualizer.hypergraph.hypergraph.number_of_edges()}")
        export_visualizations(
            visualizer,
            concept_ids,
            out_dir=out_dir,
            show_plots=not args.no_show,
            with_pipeline=args.with_pipeline,
            prefix="",
        )
        return

    if args.npy:
        concept_ids = np.load(args.npy).astype(np.float32)
        th = args.threshold if args.threshold is not None else 0.1
        out_dir = args.out or os.getcwd()
        nc = concept_ids.shape[1]
        visualizer = Stage1Visualizer(num_concepts=nc)
        visualizer.build_hypergraph(concept_ids, threshold=th)
        export_visualizations(
            visualizer,
            concept_ids,
            out_dir=out_dir,
            show_plots=not args.no_show,
            with_pipeline=args.with_pipeline,
            prefix="",
        )
        return

    run_demo()


if __name__ == "__main__":
    main()