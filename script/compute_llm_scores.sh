#############################################################
# Compute LLM scores for all models
#############################################################

# model list
MODEL_TYPE_LIST=( \
    "llama" \
    "mpt" \
    "dolly" \
    "gpt3" \
    "chatgpt" \
    "gpt4" \
)

for MODEL_TYPE in ${MODEL_TYPE_LIST[@]}; do
    for PROMPT in "zero-shot" "few-shot"; do
        echo
        echo "Computing Scores for ${MODEL_TYPE}-${PROMPT}"

        COMMAND="python src/llm_exp.py compute-score \
            --model-type ${MODEL_TYPE} \
            --test-file data/test_subset.json \
            --output-folder output/${MODEL_TYPE}-${PROMPT}"
        echo "COMMAND: ${COMMAND}"
        echo
        ${COMMAND}
    done
done
