

data_path=/home/jiawen/AppraisalBench/output/evaluation/dialog_first_given_appraisal/Qwen3-8B_Qwen3-8B_Qwen3-8B.jsonl
echo analysing $data_path ...
conda run -n appbench python scripts/analyze_base_eval.py --pred $data_path --out-dir output/analysis/base_eval