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
# limitations under the License.
"""
Preprocess the aime1983-2023 dataset to parquet format
"""

import os
import datasets
import pandas as pd
import random

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/train/aime_val2')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'gneubig/aime-1983-2024'
    # data_source = 'aime-2022-2023'
    file_name = 'aime_1983_2023.parquet'


    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    
    # 过滤掉2024年的数据（将作为测试集）
    original_size = len(train_dataset)
    # train_dataset = train_dataset.filter(lambda example: example['Year'] >= 2022 and example['Year'] <= 2023)
    train_dataset = train_dataset.filter(lambda example: example['Year'] < 2024)
    filtered_size = original_size - len(train_dataset)
    print(f"已过滤掉 {filtered_size} 条2024年的数据，剩余 {len(train_dataset)} 条数据")

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('Question')

            question = question + ' ' + instruction_following

            answer = example.pop('Answer')
            # solution = extract_solution(answer)

            tmp_data_source = 'aime' + str(example['Year'])
            data = {
                # "data_source": tmp_data_source,
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir


    train_dataset.to_parquet(os.path.join(local_dir, file_name))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
    
    # 随机读取一条数据并显示
    print("\n读取保存的parquet文件并随机显示一条数据:")
    df = pd.read_parquet(os.path.join(local_dir, file_name))
    # random_row = df.sample(n=1).iloc[0]
    random_row = df.iloc[0]
    print("随机数据样例:")
    print(random_row.to_dict())
    print(f"keys: {random_row.keys()}")

    # 输出最后一条数据
    last_row = df.iloc[-1]
    print("最后一条数据样例:")
    print(last_row.to_dict())
    print(f"keys: {last_row.keys()}")
