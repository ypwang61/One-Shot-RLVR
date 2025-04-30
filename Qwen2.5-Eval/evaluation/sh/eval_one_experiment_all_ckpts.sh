# rm -rf sh/eval_checkpoint_yiping.sh; vim sh/eval_checkpoint_yiping.sh
PROMPT_TYPE="qwen25-math-cot"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MAX_TOKENS="3072"

# CHECKPOINTS_DIR=... # replace with your own path for storing checkpoints




####### pi1 #######
PROJECT_NAME="verl_few_shot"
EXPERIMENT_NAME="Qwen2.5-Math-1.5B-pi1_r128"
GLOBAL_STEP_LIST=($(seq 20 20 2000))

# # Loop through each step in the list
for GLOBAL_STEP in "${GLOBAL_STEP_LIST[@]}"; do
    echo "======== Evaluating checkpoint at global step: ${GLOBAL_STEP} ========"
    MODEL_NAME_OR_PATH=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${GLOBAL_STEP}/actor
    OUTPUT_DIR=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/eval/global_step_${GLOBAL_STEP}
    bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MAX_TOKENS $OUTPUT_DIR
done


####### DSR-sub #######
# PROJECT_NAME="verl_few_shot"
# EXPERIMENT_NAME="Qwen2.5-Math-1.5B-dsr_sub"
# GLOBAL_STEP_LIST=($(seq 20 20 2000))

# # # Loop through each step in the list
# for GLOBAL_STEP in "${GLOBAL_STEP_LIST[@]}"; do
#     echo "======== Evaluating checkpoint at global step: ${GLOBAL_STEP} ========"
#     MODEL_NAME_OR_PATH=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${GLOBAL_STEP}/actor
#     OUTPUT_DIR=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/eval/global_step_${GLOBAL_STEP}
#     bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MAX_TOKENS $OUTPUT_DIR
# done


