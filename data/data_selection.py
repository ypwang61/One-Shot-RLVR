import json
import pandas as pd
import random 
import numpy as np
import os

import argparse


def lim(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        min_length = 1000
        for values in data.values():
            if values and len(values) < min_length:
                min_length = len(values)
        print(min_length)
        print("Datasize size: ", len(data))

        epoch_sums = [0] * min_length
        for key, values in data.items():
            for i in range(min_length):
                epoch_sums[i] += values[i]

        epoch_averages = [epoch_sum/len(data) for epoch_sum in epoch_sums]

        minus_square_sum = 0
        for i in range(min_length):
            minus_square_sum += (1-epoch_averages[i]) * (1-epoch_averages[i])
        
        result = {}
        for key, values in data.items():
            local_minus_square_sum = 0
            for i in range(min_length):
                local_minus_square_sum += (values[i]-epoch_averages[i]) * (values[i]-epoch_averages[i])
            result[key] = 1 - local_minus_square_sum/minus_square_sum

        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not a valid JSON file.")
        return {}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}

def acc_score(file_path, method='std'):

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        result = {}
        for key, values in data.items():
            if method == 'std':
                count = np.std(values)

            elif method == 'random':
                # select a random value between 0 and 1, unrelated to the values
                count = random.random()
            elif method == 'lim':
                return lim(file_path)
                
            result[key] = count

        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not a valid JSON file.")
        return {}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}

def sample_parquet_by_indices(json_file_path, parquet_file_path, output_parquet_path, 
                            index_column='index', top_n=128, top_start=None, top_end=None, 
                            method='std', repeat_time=1, is_save=1):
    if top_start is not None and top_end is not None:
        print(f"Sampling records from index {top_start} to {top_end} from {parquet_file_path} using {method} method.")
        expected_sample_count = (top_end - top_start + 1)
    else:
        print(f"Sampling {top_n} records from {parquet_file_path} using {method} method.")
        expected_sample_count = top_n
        
    sorted_counts = acc_score(json_file_path, method)
    if not sorted_counts:
        print("No valid counts obtained from the JSON file / parquet file.")
        return 0, 0
    
    # Load the original JSON data to access raw values
    with open(json_file_path, 'r') as f:
        original_json_data = json.load(f)

    string_indices = list(sorted_counts.keys())
    
    # Select indices based on range if provided, otherwise use top_n
    if top_start is not None and top_end is not None:
        string_indices = string_indices[top_start:top_end+1]
    else:
        string_indices = string_indices[:min(top_n, len(sorted_counts))]
        
    indices_to_keep = [int(idx) for idx in string_indices]
    print(indices_to_keep)

    df = pd.read_parquet(parquet_file_path, engine='pyarrow')

    result_data = []
    for _, row in df.iterrows():
        try:
            if row['extra_info']['index'] in indices_to_keep:
                result_data.append(row)
        except:
            continue

    filtered_df = pd.DataFrame(result_data)
    
    # Handle data repetition if repeat_time > 1
    if repeat_time > 1:
        # Create a list to store the repeated dataframes
        repeated_dfs = [filtered_df] * repeat_time
        # Concatenate all the repeated dataframes
        filtered_df = pd.concat(repeated_dfs, ignore_index=True)
        
        # Update the output path to include the repeat information
        total_count = expected_sample_count * repeat_time
        output_base, output_ext = os.path.splitext(output_parquet_path)
        output_parquet_path = f"{output_base}_repeatto{total_count}{output_ext}"
    
    # Randomly select up to 10 samples from the filtered dataframe
    sample_size = min(10, len(filtered_df))
    sample_indices = random.sample(range(len(filtered_df)), sample_size)
    print("\n=== Randomly selected samples ===")
    
    for i, idx in enumerate(sample_indices):
        sample_row = filtered_df.iloc[idx]
        row_index = str(sample_row['extra_info']['index'])
        
        print(f"\nSample {i+1} (Index: {row_index}):")
        print("Raw count list from JSON:")
        print(original_json_data.get(row_index, []))
        print("Sample row data:")
        print(sample_row)
        print("prompt:")
        print(sample_row['prompt'][0]['content'])
        print("-" * 80)
    
    # Count unique prompts for verification
    unique_prompts = set()
    for _, row in filtered_df.iterrows():
        prompt_content = row['prompt'][0]['content']
        unique_prompts.add(prompt_content)
    
    print(f"Sampled {len(filtered_df)} records out of {len(df)} total.")
    print(f"Number of unique prompts in the output file: {len(unique_prompts)}")
    
    # Save to new parquet file
    if is_save:
        filtered_df.to_parquet(output_parquet_path)
        print(f"Saved to {output_parquet_path}")
    else:
        print(f"Not saving to file (is_save=0). Would have saved to {output_parquet_path}")
    
    return len(filtered_df), len(unique_prompts)

def random_sample_parquet(parquet_file_path, output_parquet_path, sample_size=128, is_save=1):
    df = pd.read_parquet(parquet_file_path, engine='pyarrow')
    
    total_records = len(df)
    
    if total_records <= sample_size:
        print(f"Warning: Requested sample size ({sample_size}) is greater than or equal to "
                f"the total number of records ({total_records}). Returning all records.")
        df.to_parquet(output_parquet_path)
        return total_records
    
    random_indices = random.sample(range(total_records), sample_size)
    
    sampled_df = df.iloc[random_indices]
    
    # Save to new parquet file
    if is_save:
        sampled_df.to_parquet(output_parquet_path)
        print(f"Saved to {output_parquet_path}")
    else:
        print(f"Not saving to file (is_save=0). Would have saved to {output_parquet_path}")
    
    return sample_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample parquet files based on different methods.')
    parser.add_argument("--index_json_path", type=str, default="acc_step_500.json", help="Path to the index json file")
    parser.add_argument("--data_dir", type=str, default="data/train/one_shot_rlvr", help="Path to the data directory")
    parser.add_argument("--parquet_file_name", type=str, default="dsr_sub.parquet", help="Path to the parquet file")
    parser.add_argument('--top_n', type=int, default=1, help='Number of top records to sample')
    parser.add_argument('--repeat_time', type=int, default=128, help='Number of times to repeat the sampling')
    parser.add_argument('--top_index', type=int, default=1200, help='Index of the top record')
    parser.add_argument('--top_start', type=int, default=None, help='Start index of the range selection')
    parser.add_argument('--top_end', type=int, default=None, help='End index of the range selection')
    parser.add_argument('--method', type=str, default='std', help='Method to sample the parquet file')
    parser.add_argument('--is_save', type=int, default=1, help='Whether to save the output parquet file (1=yes, 0=no)')
    args = parser.parse_args()


    
    index_json_path = args.index_json_path
    data_dir = args.data_dir
    parquet_file_path = f"{data_dir}/{args.parquet_file_name}"

    top_n = args.top_n #417
    repeat_time = args.repeat_time
    top_index = args.top_index
    top_start = None  # Default to None when not using range selection
    top_end = None    # Default to None when not using range selection
    if top_index is not None:
        top_start = top_index
        top_end = top_index
    print(f"top_start: {top_start}, top_end: {top_end}, top_n: {top_n}")


    method_name = args.method

    # Example of using range selection (uncomment to use)
    # top_start = 100
    # top_end = 200
    
    is_save = args.is_save

    if top_start is not None and top_end is not None:
        if top_start == top_end:
            output_parquet_path = f"{data_dir}/dsr_sub_{method_name}_pi{top_start+1}_r{repeat_time}.parquet"
        else:
            output_parquet_path = f"{data_dir}/dsr_sub_{method_name}_pi{top_start+1}-{top_end+1}_r{repeat_time}.parquet"
    else:
        output_parquet_path = f"{data_dir}/dsr_sub_{method_name}_sample{top_n}_r{repeat_time}.parquet"
        
    print(acc_score(index_json_path, method=method_name))
    sample_parquet_by_indices(index_json_path, parquet_file_path, output_parquet_path, 
                            top_n=top_n, top_start=top_start, top_end=top_end,
                            method=method_name, repeat_time=repeat_time, is_save=is_save)

