# Concept Visualization

这个目录专门存放 SCAR-CBM 的 concept 可视化脚本和面板生成工具。

当前包含：

- `common.py`
  - 通用数据加载、CUB 样本索引、属性名读取、热图叠加与模型前向工具
- `generate_concept_panels.py`
  - 基于 Stage2 `.npz` 和 Stage5 权重，批量生成 concept 叠加图、before/after/delta 对比图、汇总 JSON
- `build_html_panel.py`
  - 读取 `manifest.json`，生成一个可直接打开的静态 HTML 面板

典型用法：

```bash
python concept_visualization/generate_concept_panels.py \
  --cub-root Data/CUB_200_2011 \
  --attributes Data/attributes.txt \
  --stage1-out Stage1/stage1_cub_output_rerun_260409 \
  --stage2-npz Stage2_Semantic-feature-extraction/stage2_output_rerun_260409/stage2_features.npz \
  --stage4-npz Stage4_Anti-curriculum/stage4_output_rerun_260409/stage4_sampled_indices.npz \
  --stage5-weights Stage5_Graph-logic_Optimization/stage5_output_rerun_260409/model_final.pt \
  --sample-count 8 \
  --top-k 6 \
  --out-dir concept_visualization/output_rerun_260409

python concept_visualization/build_html_panel.py \
  --manifest concept_visualization/output_rerun_260409/manifest.json \
  --out concept_visualization/output_rerun_260409/index.html
```
