# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts and evaluate checkpoints from all steps
"""
import csv
import ray
import numpy as np
import hydra
import os
import glob
import re
import matplotlib.pyplot as plt
from tabulate import tabulate

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


def process_single_model(config, model_path):
    """Process a single model checkpoint"""
    print(f"Processing model: {model_path}")
    
    # Store original model path to restore later
    original_model_path = config.model.path
    
    # Override the model path
    config.model.path = model_path
    
    # Generate a unique identifier for this run to avoid conflicts
    import uuid
    run_id = uuid.uuid4().hex[:8]
    
    # We'll use a monkey patch approach instead of modifying the config directly
    # Store the original methods that we'll need to patch
    original_get_placement_groups = RayResourcePool.get_placement_groups
    
    # Determine output path based on checkpoint
    step_match = re.search(r'global_step_(\d+)', model_path)
    step_num = step_match.group(1) if step_match else "unknown"
    
    output_dir = os.path.dirname(config.data.output_path)
    step_output_path = os.path.join(output_dir, f"output_step_{step_num}.parquet")
    original_output_path = config.data.output_path
    config.data.output_path = step_output_path
    
    # Read dataset
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    print(local_path)
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Monkey patch the get_placement_groups method to use our unique ID
    def patched_get_placement_groups(self, strategy=None, lifetime="detached"):
        # The original method doesn't accept pg_name_prefix, so we'll modify its behavior
        # without changing its signature
        original_pgs = original_get_placement_groups(self, strategy, lifetime)
        
        # Modify the placement group names in a different way
        # Set the VERL_PG_PREFIX environment variable which might be used internally
        import os
        os.environ["VERL_PG_PREFIX"] = f"verl_group_{run_id}"
        
        return original_pgs
    
    # Apply the monkey patch
    RayResourcePool.get_placement_groups = patched_get_placement_groups

    # Initialize ray components with the patched methods
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    num_batch = (total_samples // config_batch_size) + 1
    output_lst = []  # We'll reshape at the end

    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
        
        # Skip processing if the batch is empty
        if not batch_chat_lst:
            print(f'[{batch_idx+1}/{num_batch}] Batch is empty, skipping.')
            continue
        
        # Repeat the batch n_samples times
        repeated_chat_lst = []
        for chat in batch_chat_lst:
            repeated_chat_lst.extend([chat] * config.data.n_samples)
        
        inputs = tokenizer.apply_chat_template(repeated_chat_lst,
                                               add_generation_prompt=True,
                                               padding=True,
                                               truncation=True,
                                               max_length=config.rollout.prompt_length,
                                               return_tensors='pt',
                                               return_dict=True,
                                               tokenize=True)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]
        
        if real_batch_size % dp_size != 0:
            dummy_data_size = dp_size - real_batch_size % dp_size
            dummy_data = data[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        
        # Generate all samples at once
        print(len(data.batch['input_ids']))
        output = wg.generate_sequences(data)
        # Remove dummy data
        output = output[:real_batch_size]
        output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                           skip_special_tokens=False)

        # Remove padding
        pad_token = tokenizer.pad_token
        output_text_unpad = []
        for text in output_text:
            output_text_unpad.append(text.replace(pad_token, ''))

        output_lst.extend(output_text_unpad)

    # Reshape output_lst from (total_samples,) to (n_data, n_samples)
    total_samples = len(output_lst)
    n_data = total_samples // config.data.n_samples
    output_lst = np.array(output_lst).reshape(n_data, config.data.n_samples).tolist()

    # Add to the data frame
    dataset['responses'] = output_lst

    # Write to the step-specific parquet
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(step_output_path)
    
    # Compute evaluation metrics
    prompts = dataset[config.data.prompt_key]
    responses = dataset['responses']  # Using the generated responses
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    total = len(dataset)
    total_scores = []
    
    for i in range(total):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        score_lst = []
        for r in response_lst:
            if data_source in ['deepscaler', 'aime', 'amc', 'math', 'minerva', 'olympiad']:
                if 'qwen' in config.model.path.lower() or 'llama' in config.model.path.lower():
                    score = reward_fn("", r, ground_truth)
                else:
                    score = reward_fn("", r, ground_truth, None, True)
            else:
                score = reward_fn(r, ground_truth)
            score_lst.append(score)
        max_score = np.max(score_lst)
        total_scores.append(score_lst)
        if max_score == 1:
            passes += 1

    n_samples = config.data.n_samples
    pass_at_n = passes / total
    pass_at_1 = np.mean([scores[0] for scores in total_scores])  # First sample for pass@1
    pass_average = np.mean(total_scores)

    # Properly clean up Ray resources
    try:
        wg.shutdown()
    except Exception as e:
        print(f"Warning: Error during worker group shutdown: {e}")
    
    # Clean up Ray completely
    if ray.is_initialized():
        ray.shutdown()
    
    # Restore the original configurations
    config.model.path = original_model_path
    config.data.output_path = original_output_path
    
    return {
        'step': step_num,
        'model_path': model_path,
        'pass@1': pass_at_1,
        f'pass@{n_samples}': pass_at_n,
        'pass_average': pass_average
    }


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    
    # Ensure clean start
    if ray.is_initialized():
        ray.shutdown()
    
    # Process checkpoint selection arguments
    start_step = 10
    end_step =  float('inf')
    step_gap = 10

    if hasattr(config, 'checkpoint'):
        if hasattr(config.checkpoint, 'start') and config.checkpoint.start is not None:
            start_step = config.checkpoint.start
        if hasattr(config.checkpoint, 'end') and config.checkpoint.end is not None:
            end_step = config.checkpoint.end
        if hasattr(config.checkpoint, 'step') and config.checkpoint.step is not None:
            step_gap = config.checkpoint.step
    
    # Get the base checkpoint directory path
    checkpoint_base_dir = config.model.path
    if checkpoint_base_dir.endswith('/'):
        checkpoint_base_dir = checkpoint_base_dir[:-1]
    
    # Find all checkpoint directories with actor models
    checkpoint_pattern = os.path.join(checkpoint_base_dir, "global_step_*/actor")
    all_checkpoints = glob.glob(checkpoint_pattern)
    
    # Extract step numbers and sort
    checkpoint_info = []
    for cp in all_checkpoints:
        step_match = re.search(r'global_step_(\d+)', cp)
        if step_match:
            step_num = int(step_match.group(1))
            checkpoint_info.append((step_num, cp))
    
    # Sort by step number
    checkpoint_info.sort(key=lambda x: x[0])
    
    # Filter based on start, end, and gap
    filtered_checkpoints = []
    for step_num, checkpoint_path in checkpoint_info:
        if start_step <= step_num <= end_step and (step_num - start_step) % step_gap == 0:
            filtered_checkpoints.append((step_num, checkpoint_path))
    
    if not filtered_checkpoints:
        print(f"No checkpoints found matching criteria: start={start_step}, end={end_step}, gap={step_gap}")
        return
    
    print(f"Found {len(filtered_checkpoints)} checkpoints to process after filtering:")
    
    # Process each checkpoint
    results = []
    for i, (step_num, checkpoint_dir) in enumerate(filtered_checkpoints):
        print(f"\n[{i+1}/{len(filtered_checkpoints)}] Processing checkpoint: {checkpoint_dir} (Step {step_num})")
        try:
            result = process_single_model(config, checkpoint_dir)
            results.append(result)
            print(f"Successfully processed checkpoint: {checkpoint_dir}")
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_dir}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with the next checkpoint rather than failing entirely
    
    # Sort results by step number
    results.sort(key=lambda x: int(x['step']) if x['step'].isdigit() else float('inf'))
    
    # Save aggregated metrics to CSV
    output_dir = os.path.dirname(config.data.output_path)
    csv_path = os.path.join(output_dir, 'all_passes.csv')
    
    # Write to CSV
    with open(csv_path, mode='w', newline='') as f:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Print table of results
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    # Create plots
    plot_results(results, output_dir, config.data.n_samples)
    
    # Clean up Ray at the end
    try:
        ray.shutdown()
        print("Ray resources successfully shut down")
    except Exception as e:
        print(f"Warning: Error during Ray shutdown: {e}")


def plot_results(results, output_dir, n_samples):
    """Create plots of the pass@1, pass@n, and pass_average metrics."""
    # Filter and sort results by step number
    valid_results = [r for r in results if r['step'].isdigit()]
    valid_results.sort(key=lambda x: int(x['step']))
    
    steps = [int(r['step']) for r in valid_results]
    pass_at_1 = [r['pass@1'] for r in valid_results]
    pass_at_n = [r[f'pass@{n_samples}'] for r in valid_results]
    pass_average = [r['pass_average'] for r in valid_results]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot pass@1
    ax1.plot(steps, pass_at_1, 'b-o', linewidth=2)
    ax1.set_title(f'Pass@1 vs Training Steps')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Pass@1')
    ax1.grid(True)
    
    # Plot pass@n
    ax2.plot(steps, pass_at_n, 'r-o', linewidth=2)
    ax2.set_title(f'Pass@{n_samples} vs Training Steps')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel(f'Pass@{n_samples}')
    ax2.grid(True)
    
    # Plot pass_average
    ax3.plot(steps, pass_average, 'g-o', linewidth=2)
    ax3.set_title('Pass Average vs Training Steps')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Pass Average')
    ax3.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'pass_metrics_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Create a combined plot
    plt.figure(figsize=(18, 10))
    plt.plot(steps, pass_at_1, 'b-o', linewidth=2, label='Pass@1')
    plt.plot(steps, pass_at_n, 'r-o', linewidth=2, label=f'Pass@{n_samples}')
    plt.plot(steps, pass_average, 'g-o', linewidth=2, label='Pass Average')
    plt.title(f'Pass Metrics vs Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Pass Rate')
    plt.grid(True)
    plt.legend()
    
    # Add data labels to points
    for i, (x, y1, y2, y3) in enumerate(zip(steps, pass_at_1, pass_at_n, pass_average)):
        plt.annotate(f"{y1:.3f}", (x, y1), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
        plt.annotate(f"{y2:.3f}", (x, y2), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=8)
        plt.annotate(f"{y3:.3f}", (x, y3), textcoords="offset points", 
                    xytext=(-15,0), ha='right', fontsize=8)
    
    # Save the combined plot
    combined_plot_path = os.path.join(output_dir, 'combined_pass_metrics_plot.png')
    plt.savefig(combined_plot_path)
    print(f"Combined plot saved to {combined_plot_path}")


# Add the select_reward_fn from main_eval.py
def select_reward_fn(data_source):
    if data_source == 'openai/gsm8k':
        from verl.utils.reward_score import gsm8k
        return gsm8k.compute_score
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval', 'simplerl/math500', 'simplerl/math_level3to5', 'simplerl/aime24', 'Maxwell-Jia/AIME_2024']:
        from verl.utils.reward_score import math
        return math.compute_score
    elif data_source in ['deepscaler', 'aime', 'amc', 'math', 'minerva', 'olympiad']:
        from verl.utils.reward_score import deepscaler
        return deepscaler.compute_score
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()