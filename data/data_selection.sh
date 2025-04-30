
# get pi_{i} -> top_index = i-1
python data_selection.py \
    --index_json_path acc_step_500.json \
    --data_dir train/one_shot_rlvr \
    --parquet_file_name dsr_sub.parquet \
    --repeat_time 128 \
    --top_index 0 \
    --method std \
    --top_n 0