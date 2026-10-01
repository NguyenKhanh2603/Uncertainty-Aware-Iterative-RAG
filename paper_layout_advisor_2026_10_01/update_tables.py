"""Regenerate complete advisor tables from the supplied numerical results.

Run: python paper_layout_advisor_2026_10_01/update_tables.py
Ranks use distinct displayed values: bold for best, HTML ins for second best.
Ties share the same highlighting. Missing metrics are never imputed.
"""

from pathlib import Path
import json


FOLDER = Path(__file__).resolve().parent
SOURCE = FOLDER.parent / "paper_assets/layout_results_supplied_2026_10_01.json"
METRICS = [
    ("selection", "kept_chunks", 2, False),
    ("selection", "precision_percent", 1, True),
    ("selection", "recall_percent", 1, True),
    ("selection", "selection_f1_percent", 1, True),
    ("generation", "em_percent", 2, True),
    ("generation", "f1_percent", 2, True),
    ("generation", "latency_seconds", 2, False),
    ("generation", "approx_tflops", 2, False),
]


def display(value, digits, ranked):
    if value is None:
        return "—"
    number = round(value, digits)
    formatted = f"{number:.{digits}f}"
    if number == ranked[0]:
        return f"**{formatted}**"
    if len(ranked) > 1 and number == ranked[1]:
        return f"<ins>{formatted}</ins>"
    return formatted


def dataset_table(records, dataset):
    ranks = []
    for group, key, digits, higher_better in METRICS:
        values = {
            round(record["datasets"][dataset][group][key], digits)
            for record in records if group in record["datasets"][dataset]
        }
        ranks.append(sorted(values, reverse=higher_better))
    lines = [
        "| Method | Kept ↓ | Support P (%) ↑ | Support R (%) ↑ | Selection F1 (%) ↑ | EM (%) ↑ | Downstream F1 (%) ↑ | Latency (s) ↓ | Approx. TFLOPs ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        cells = [record["method"]]
        for (group, key, digits, _), ranked in zip(METRICS, ranks):
            value = record["datasets"][dataset].get(group, {}).get(key)
            cells.append(display(value, digits, ranked))
        lines.append("| " + " | ".join(cells) + " |")
    assert len(lines) - 2 == 15
    return "\n".join(lines)


def overall_table(records):
    keys = ["em_percent", "f1_percent"]
    ranks = [sorted({r["overall"][key] for r in records if "overall" in r}, reverse=True) for key in keys]
    lines = ["| Method | Overall EM (%) ↑ | Overall downstream F1 (%) ↑ |", "|---|---:|---:|"]
    for record in records:
        cells = [record["method"]] + [
            display(record.get("overall", {}).get(key), 2, ranked)
            for key, ranked in zip(keys, ranks)
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    data = json.loads(SOURCE.read_text())
    records = data["records"]
    assert len(records) == 15
    path = FOLDER / "LAYOUT.md"
    markdown = path.read_text()
    start = markdown.index("**Cách đọc bảng:**") if "**Cách đọc bảng:**" in markdown else markdown.index("### 8.1.")
    end = markdown.index("### 8.6. Diễn giải số hiện tại") if "### 8.6." in markdown else markdown.index("### 8.3. Diễn giải số hiện tại")
    tables = (
        "**Cách đọc bảng:** mỗi dataset có toàn bộ 15 method/configurations trong bảng số hiện tại. "
        "**In đậm: tốt nhất**; <ins>gạch chân: tốt nhì</ins>, xếp theo hai giá trị khác nhau của từng cột trong cùng dataset; các hàng bằng nhau được đánh dấu cùng hạng. "
        "↑ là cao hơn tốt hơn; ↓ là thấp hơn về context/cost. Kept/cost cần đọc cùng recall và downstream. "
        "P/R/Selection F1 đo support chunks; EM/downstream F1 đo câu trả lời. **—** là số chưa có trong bảng nguồn.\n\n"
    )
    for index, dataset in enumerate(data["datasets"], 1):
        tables += f"### 8.{index}. {dataset}: toàn bộ kết quả hiện tại\n\n" + dataset_table(records, dataset) + "\n\n"
    tables += "### 8.5. Overall downstream\n\n" + overall_table(records) + "\n\n"
    tables += (
        "- Overall giữ số được báo trong bảng cung cấp. F1 của Two-Stage `α₂=.40` được báo là **46.95%**; "
        "trung bình bốn ô dataset đã làm tròn là 46.9425%. Số tổng hợp sẽ được đối chiếu với log chưa làm tròn.\n"
        "- Bảng còn thiếu downstream cho Stage 1 `.90` và selection cho Two-Stage `α₂=.40`; "
        "hai cấu hình vẫn được giữ đủ hàng, với các ô tương ứng là **—**.\n"
        "- Dữ liệu bảng: [các số từ bản LaTeX cập nhật](../paper_assets/layout_results_supplied_2026_10_01.json).\n\n"
    )
    markdown = markdown[:start] + tables + markdown[end:]
    markdown = markdown.replace("### 8.3. Diễn giải số hiện tại", "### 8.6. Diễn giải số hiện tại", 1)
    heading = "## 9. Đang hoàn thiện để cải thiện kết quả\n\n"
    note = "**Đang làm việc cùng anh Hưng để cải thiện phương pháp**, tập trung bảo toàn evidence khi prune và cải thiện đồng thời chất lượng chọn chunks, downstream và chi phí.\n\n"
    if note not in markdown:
        assert heading in markdown
        markdown = markdown.replace(heading, heading + note, 1)
    path.write_text(markdown)
    print("Updated 4 × 15 dataset rows and 15 Overall rows; best/second-best ranked per metric.")


if __name__ == "__main__":
    main()
