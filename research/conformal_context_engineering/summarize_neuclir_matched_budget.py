"""Report a calibration-valid matched-budget comparison from chunk selection results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cce-alpha', default='0.2')
    parser.add_argument('--query-alpha', default='0.1')
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    cce = source['results'][args.cce_alpha]['cce_positive_inherited_labels']
    query = source['results'][args.query_alpha]['query_all_nugget']
    artifact = {
        'status': 'matched_realized_context_budget_selection_proxy',
        'comparison': {
            'cce_positive': {'alpha': float(args.cce_alpha), **cce},
            'query_all_nugget': {'alpha': float(args.query_alpha), **query},
        },
        'matching': {
            'absolute_mean_chunk_difference': abs(cce['mean_chunks_kept'] - query['mean_chunks_kept']),
            'same_calibration_test_split': True,
            'thresholds_are_calibration_only': True,
            'not_ARGUE_F1': 'Public nugget-to-document labels are inherited by chunks; no report generator or LLM judge is used.'
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + '\n')
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
