@echo off
set PYTHONPATH=src
echo Preparing Pooled Reference Banks...
python scripts\prepare_conformal_reference_banks.py --input "E:\Downloads\conformal_backfill_2_3_results\mmqa_top30_retrieval.jsonl.gz" "E:\Downloads\conformal_backfill_2_3_results\webqa_top30_retrieval.jsonl.gz" "E:\Downloads\conformal_backfill_2_3_results\hotpotqa_top30_retrieval.jsonl.gz" "E:\Downloads\conformal_backfill_2_3_results\tatqa_top30_retrieval.jsonl.gz" --output "E:\Downloads\conformal_backfill_2_3_results\pooled_reference_banks.json.gz" --conditioning "dataset,modality"
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running Selection...
python scripts\run_conformal_backfill_selection.py --input "E:\Downloads\conformal_backfill_2_3_results\mmqa_top30_retrieval.jsonl.gz" "E:\Downloads\conformal_backfill_2_3_results\webqa_top30_retrieval.jsonl.gz" "E:\Downloads\conformal_backfill_2_3_results\hotpotqa_top30_retrieval.jsonl.gz" "E:\Downloads\conformal_backfill_2_3_results\tatqa_top30_retrieval.jsonl.gz" --bank "E:\Downloads\conformal_backfill_2_3_results\pooled_reference_banks.json.gz" --output-dir "E:\Downloads\conformal_backfill_2_3_results\pooled_bh_selection" --split-role development
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running Benchmark...
python scripts\benchmark_conformal_backfill.py --input "E:\Downloads\conformal_backfill_2_3_results\pooled_bh_selection\selection_decisions.jsonl.gz" --output-dir "E:\Downloads\conformal_backfill_2_3_results\pooled_bh_benchmark" --alphas 0.01,0.025,0.05,0.10,0.20,0.30,0.50,0.75,0.90 --max-context 10
echo Done!
