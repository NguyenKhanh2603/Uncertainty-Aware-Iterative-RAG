from __future__ import annotations
import json
import gzip
from pathlib import Path
from typing import Optional

from eval.datasets.loader import DatasetLoader, EvalExample
from uncertainty_rag.modality.base import ContextChunk, ModalityHandler
from uncertainty_rag.modality.multimodal_handler import MultimodalHandler

class MultiModalQALoader(DatasetLoader):
    def __init__(self, data_dir: str = "data/multimodalqa"):
        self.data_dir = Path(data_dir)
        self.tables = {}
        self.texts = {}
        self.images = {}
        self.loaded = False

    @property
    def name(self) -> str:
        return "multimodalqa"

    def _load_assets(self):
        if self.loaded: return
        for t, d_dict in [("texts", self.texts), ("tables", self.tables), ("images", self.images)]:
            p = self.data_dir / f"MMQA_{t}.jsonl.gz"
            if p.exists():
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        if t == "tables":
                            d_dict[data["id"]] = data
                        elif t == "images":
                            d_dict[data["id"]] = data["path"]
                        else:
                            d_dict[data["id"]] = data["text"]
        self.loaded = True

    def _convert_table(self, table_data: dict) -> str:
        try:
            headers = [c["column_name"] for c in table_data.get("table", {}).get("header", [])]
            rows = table_data.get("table", {}).get("table_rows", [])
            md = f"| {' | '.join(headers)} |\n|{'|'.join(['---']*len(headers))}|\n"
            for row in rows:
                md += f"| {' | '.join([c.get('text', '') for c in row])} |\n"
            return md
        except:
            return str(table_data)

    def load(self, split: str = "dev", max_examples: Optional[int] = None) -> list[EvalExample]:
        if split == "validation":
            split = "dev"
        self._load_assets()
        qa_file = self.data_dir / f"MMQA_{split}.jsonl.gz"
        examples, idx = [], 0
        with gzip.open(qa_file, "rt", encoding="utf-8") as f:
            for line in f:
                if max_examples and idx >= max_examples: break
                entry = json.loads(line)
                meta = entry.get("metadata", {})
                chunks = []
                
                # Texts
                for tid in meta.get("text_doc_ids", []):
                    if tid in self.texts:
                        chunks.append(ContextChunk(tid, self.texts[tid], "text", {"source": "paragraph"}))
                
                # Table
                tab_id = meta.get("table_id")
                if tab_id and tab_id in self.tables:
                    chunks.append(ContextChunk(tab_id, self._convert_table(self.tables[tab_id]), "table", {"source": "table"}))
                
                # Images
                for iid in meta.get("image_doc_ids", []):
                    if iid in self.images:
                        img_path = str(self.data_dir / "final_dataset_images" / self.images[iid])
                        chunks.append(ContextChunk(iid, img_path, "image", {"source": "image"}))

                examples.append(EvalExample(
                    query_id=entry.get("qid", f"mmqa_{idx}"),
                    query=entry.get("question", ""),
                    gold_answers=[str(a) for a in entry.get("answers", [])],
                    context_chunks=chunks,
                    modality="multimodal",
                    metadata={"type": meta.get("type", ""), "modalities": meta.get("modalities", [])}
                ))
                idx += 1
        return examples

    def get_modality_handler(self) -> ModalityHandler:
        return MultimodalHandler()

    def get_metrics(self) -> list[str]:
        return ["em", "f1"]
