import json
data = json.load(open(r'E:\Downloads\conformal prediction\Conformal\Uncertainty-Aware-Iterative-RAG\conformal_backfill_2_3_gpu_colab.ipynb', encoding='utf-8'))
for i, cell in enumerate(data.get('cells', [])):
    source = "".join(cell.get('source', []))
    if 'EM' in source or 'F1' in source or 'groundtruth' in source or 'evaluate' in source.lower():
        print(f"--- Cell {i} ---")
        print(source[:500])
