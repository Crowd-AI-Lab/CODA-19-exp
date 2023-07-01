############################################################
# Train CODA19 + Position-Encoding model with SciBERT
############################################################
DATA="coda19-position"
MODEL="allenai/scibert_scivocab_uncased"

DATA_FOLDER="data/${DATA}"
OUTPUT_FOLDER="model/${DATA##*/}-${MODEL##*/}"
MODEL_NAME="${MODEL}"

COMMAND="sh script/train_arg.sh \
  ${DATA_FOLDER} \
  ${OUTPUT_FOLDER} \
  ${MODEL_NAME} \
  ${DEVICE}"

echo "COMMAND: ${COMMAND}"
${COMMAND}