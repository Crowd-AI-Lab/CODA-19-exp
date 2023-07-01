#############################################################
# Fine-tune all the models using different datasets
#############################################################
MODEL="allenai/scibert_scivocab_uncased"
DEVICE=0
DATA_LIST=( \
    "coda19" \
    "coda19-position" \
    "pubmed" \
    "pubmed-position" \
    "simple-mix-position" \
    "upsampling-mix-position" \
    "pubmed-position-coda19-label" \
)

for DATA in ${DATA_LIST[@]}; do
    DATA_FOLDER="data/${DATA}"
    OUTPUT_FOLDER="model/${DATA##*/}-${MODEL##*/}"
    MODEL_NAME="${MODEL}"

    COMMAND="sh script/train_arg.sh \
      ${DATA_FOLDER} \
      ${OUTPUT_FOLDER} \
      ${MODEL_NAME} \
      ${DEVICE}"
    
    echo
    echo "COMMAND: ${COMMAND}"
    ${COMMAND}
    echo
done

# For two-stage fine-tuning, we need to further fine-tune the "pubmed-position-coda19-label" model using the "coda19-position" dataset.
DATA_FOLDER="data/coda19-position"
OUTPUT_FOLDER="model/two-stage-mix-position"
MODEL_NAME="model/pubmed-position-coda19-label-scibert_scivocab_uncased"

COMMAND="sh script/train_arg.sh \
  ${DATA_FOLDER} \
  ${OUTPUT_FOLDER} \
  ${MODEL_NAME} \
  ${DEVICE}"

echo
echo "COMMAND: ${COMMAND}"
${COMMAND}
echo

