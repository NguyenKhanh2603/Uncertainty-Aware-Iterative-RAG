@echo off
set PYTHONPATH=src
echo Preparing Rank-Binned Reference Banks...
python scripts\prepare_conformal_reference_banks.py --input "E:\Downloads\conformal_backfill_2_3_results\mmqa_top30_retrieval.jsonl.gz" "E:\Downloads\conformal_backfill_2_3_results\webqa_top30_retrieval.jsonl.gz" "E:\Downloads\conformal_backfill_2_3_results\hotpotqa_top30_retrieval.jsonl.gz" "E:\Downloads\conformal_backfill_2_3_results\tatqa_top30_retrieval.jsonl.gz" --output "E:\Downloads\conformal_backfill_2_3_results\rank_binned_reference_banks.json.gz" --conditioning "dataset,modality,rank_bin" --rank-bins "1-3,4-10,11-20,21-30" --allow-small-banks
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running Selection...
python scripts\run_conformal_backfill_selection.py --input "E:\Downloads\conformal_backfill_2_3_results\mmqa_top30_retrieval.jsonl.gz" "E:\Downloads\conformal_backfill_2_3_results\webqa_top30_retrieval.jsonl.gz" "E:\Downloads\conformal_backfill_2_3_results\hotpotqa_top30_retrieval.jsonl.gz" "E:\Downloads\conformal_backfill_2_3_results\tatqa_top30_retrieval.jsonl.gz" --bank "E:\Downloads\conformal_backfill_2_3_results\rank_binned_reference_banks.json.gz" --output-dir "E:\Downloads\conformal_backfill_2_3_results\rank_binned_selection" --split-role development --allow-underpowered-banks
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running Benchmark...
python scripts\benchmark_conformal_backfill.py --input "E:\Downloads\conformal_backfill_2_3_results\rank_binned_selection\selection_decisions.jsonl.gz" --output-dir "E:\Downloads\conformal_backfill_2_3_results\rank_binned_benchmark" --alphas 0.01,0.025,0.05,0.10,0.20,0.30,0.50 --max-context 10
echo Done!
