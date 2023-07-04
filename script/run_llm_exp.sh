#############################################################
# Predict with zero-shot and few-shot settings using LLMs
#############################################################

# model list
MODEL_LIST=( \
    "/data/data/share/llama/llama-7B"
    "mosaicml/mpt-7b-instruct" \
    "databricks/dolly-v2-3b" \
    "None" \
    "None" \
    "None" \
)

MODEL_TYPE_LIST=( \
    "llama" \
    "mpt" \
    "dolly" \
    "gpt3" \
    "chatgpt" \
    "gpt4" \
)

# run the experiment
for ((i=0;i<${#MODEL_LIST[@]};++i)); do
    MODEL=${MODEL_LIST[i]}
    MODEL_TYPE=${MODEL_TYPE_LIST[i]}
    echo
    echo "Running experiment for ${MODEL_TYPE} model: ${MODEL}"

    for PROMPT in "zero-shot.txt" "few-shot.txt"; do
        echo "Running experiment for prompt: ${PROMPT}"

        COMMAND="python src/llm_exp.py exp \
            --model-path ${MODEL} \
            --model-type ${MODEL_TYPE} \
            --test-file data/test_subset.json \
            --prompt-file prompt/${PROMPT} \
            --output-folder output/${MODEL_TYPE}-${PROMPT%.*}"

        # echo command and run
        echo "COMMAND: ${COMMAND}"
        echo
        ${COMMAND}
    done
done