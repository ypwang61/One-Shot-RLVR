<h1 style="text-align: center;">Reinforcement Learning for Reasoning in Large Language Models with One Training Example</h1>

<div style="text-align: center;">
<a href="https://ypwang61.github.io/">Yiping Wang</a>, <a href="https://www.linkedin.com/in/qing-yang-b3a02120b/">Qing Yang</a>, <a href="https://zhiyuan-zeng.github.io/">Zhiyuan Zeng</a>, <a href="https://renll.github.io/">Liliang Ren</a>, <a href="https://liyuanlucasliu.github.io/">Lucas Liu</a>, <a href="https://www.microsoft.com/en-us/research/people/baolinpeng/">Baolin Peng</a>, <a href="https://www.microsoft.com/en-us/research/people/chehao/">Hao Cheng</a>, <a href="https://sheehan1230.github.io/">Xuehai He</a>, <a href="https://github.com/kuan-wang">Kuan Wang</a>, <a href="https://www.microsoft.com/en-us/research/people/jfgao/">Jianfeng Gao</a>, <a href="https://www.microsoft.com/en-us/research/people/wzchen/">Weizhu Chen</a>, <a href="https://www.microsoft.com/en-us/research/people/shuowa/">Shuohang Wang</a>, <a href="https://simonshaoleidu.com/">Simon Shaolei Du</a>, <a href="https://www.linkedin.com/in/yelong-shen-84b0122b/">Yelong Shen</a>
<p>

<a href="https://arxiv.org/abs/2504.20571"><img src="https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
</p>
</div>

## Updates
* 30/04/2025: 🎉 We release our [paper](https://arxiv.org/abs/2504.20571), [code](https://github.com/ypwang61/One-Shot-RLVR), and [wandb records](https://wandb.ai/yipingwanguw/verl_few_shot?nw=nwuseryipingwang22).


## Setup


### Train Enviroment
Our training pipeline is adapted from [verl](https://github.com/volcengine/verl) and  [rllm(DeepScaleR)](https://github.com/agentica-project/rllm). The installation commands that we verified as viable are as follows:
```bash
conda create -y -n rlvr_train python=3.10
conda activate rlvr_train
pip install -e .
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install ray vllm==0.6.3
pip install flash-attn --no-build-isolation
pip install wandb matplotlib
pip install huggingface_hub
```
### Eval Enviroment
Our evaluation pipeline for math reasoning tasks is adapted from [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math). The installation commands that we verified as viable are as follows:
```bash
conda create -y -n rlvr_eval python=3.10
conda activate rlvr_eval
cd Qwen2.5-Eval/evaluation
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3
pip install wandb matplotlib
pip install -U transformers
pip install vllm==0.6.3
```


## Data
### DSR-sub
We randomly select a subset consisting of 1209 examples from [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) (DSR-sub), and we use it as the instance pool for data selection. We include the training example used in our paper in `data/train/one_shot_rlvr`. For 1(few)-shot RLVR dataset, we duplicate the data until training batch size (in our experiment it is 128). 



(Optionally) To obtain the training example, we rank DSR-sub by the historical variance score, which calculates the variance of the historical accuracy (We hope this can inspire better data selection way in the future). To obtain examples $\pi_i$ based on the historical accuracy of Qwen2.5-Math-1.5B, we can change the `top_index` parameter in `data/data_selection.sh` to $i-1$, and run then run `bash data_selection.sh`.


As a reference, we present example $\pi_1$ here: 
<!-- and $\pi_{13}$ as follows. -->

#### $\pi_1$:
```text
Prompt:
"The pressure \\( P \\) exerted by wind on a sail varies jointly as the area \\( A \\) of the sail and the cube of the wind's velocity \\( V \\). When the velocity is \\( 8 \\) miles per hour, the pressure on a sail of \\( 2 \\) square feet is \\( 4 \\) pounds. Find the wind velocity when the pressure on \\( 4 \\) square feet of sail is \\( 32 \\) pounds. Let's think step by step and output the final answer within \\boxed{}."

Ground truth (label in DSR-sub):
12.8.
```

<!-- #### $\pi_{13}$:
```text
Prompt:
"Given that circle $C$ passes through points $P(0,-4)$, $Q(2,0)$, and $R(3,-1)$.  \n$(1)$ Find the equation of circle $C$.  \n$(2)$ If the line $l: mx+y-1=0$ intersects circle $C$ at points $A$ and $B$, and $|AB|=4$, find the value of $m$. Let's think step by step and output the final answer within \\boxed{}."

Ground truth (label in DSR-sub):
\frac{4}{3}.
``` -->


## Training
Before training, we can assign the checkpoint path:
```bash
export CHECKPOINTS_DIR=./checkpoints/ # your checkpoint path
```

To run 1-shot RLVR with $\pi_1$, we can run:
```bash
conda activate rlvr_train
bash scripts/train/training_1.5b_pi1_r128.sh
```

As a comparison, the commands for running full-set RLVR on DSR-sub is as below:
```bash
conda activate rlvr_train
bash scripts/train/training_1.5b_dsr_sub.sh 
```

Please change `data.train_files` and `trainer.experiment_name` in the training script when trying other training examples.

## Evaluation

### Eval Scripts
To run evaluation for 1-shot RLVR with $\pi_1$ on 6 common math reasoning benchmarks (MATH500, AIME24, AMC23, Minerva Math, OlympiadBench, AIME25), we can follow the commands:
```bash
conda activate rlvr_eval
cd Qwen2.5-Eval/evaluation
bash sh/eval_one_experiment_all_ckpts.sh
```
Here for AIME24, AMC23, and AIME25, we evaluate the pass@8 results.
Please adjust the experiment name in `Qwen2.5-Eval/evaluation/sh/eval_one_experiment_all_ckpts.sh` when using other training examples. 


## W&B
We have logged our experiments for three models to [this wandb project](https://wandb.ai/yipingwanguw/verl_few_shot?nw=nwuseryipingwang22), including the results of 1(few)-shot RLVR on [`Qwen2.5-Math-1.5B`](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B), [`Qwen2.5-Math-7B`](https://huggingface.co/Qwen/Qwen2.5-Math-7B) and [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B). We also include the baseline of the full-set RLVR with DSR-sub in it. Please note that the validation results displayed are calculated using the verl/rllm framework and may differ slightly from qwen-eval results.

## Acknowledgements
- Our training experiments are powered by a modified fork of [rllm(DeepScaleR)](https://github.com/agentica-project/rllm) and [verl](https://github.com/volcengine/verl).
- Our evaluation experiments are based on a modified fork of [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math).
- Our model is trained on top of [`Qwen2.5-Math-1.5B`](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B), [`Qwen2.5-Math-7B`](https://huggingface.co/Qwen/Qwen2.5-Math-7B), [`Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).

  
## Citation
```bibtex
@article{wang2025reinforcement,
  title={Reinforcement Learning for Reasoning in Large Language Models with One Training Example},
  author={Wang, Yiping and Yang, Qing and Zeng, Zhiyuan and Ren, Liliang and Liu, Lucas and Peng, Baolin and Cheng, Hao and He, Xuehai and Wang, Kuan and Gao, Jianfeng and Chen, Weizhu and Wang, Shuohang and Du, Simon Shaolei and Shen, Yelong},
  journal={arXiv preprint arXiv:2504.20571},
  year={2025}
}
```
