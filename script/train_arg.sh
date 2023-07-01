# Take arg as input
DATA_FOLDER="$1"
OUTPUT_FOLDER="$2"
MODEL_NAME="$3"

# Take $4 as the input for cuda device and the default is 0
CUDA_VISIBLE_DEVICES="${4:-0}"

# Echo
echo
echo "         DATA_FOLDER: ${DATA_FOLDER}"
echo "       OUTPUT_FOLDER: ${OUTPUT_FOLDER}"
echo "          MODEL_NAME: ${MODEL_NAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python src/run_glue.py \
  --model_name_or_path ${MODEL_NAME} \
  --train_file "${DATA_FOLDER}/train.json" \
  --validation_file "${DATA_FOLDER}/dev.json" \
  --test_file "${DATA_FOLDER}/test.json" \
  --do_train \
  --do_eval \
  --do_predict \
  --pad_to_max_length false \
  --max_seq_length 512 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --auto_find_batch_size \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --warmup_ratio 0.1 \
  --fp16 true \
  --output_dir "${OUTPUT_FOLDER}" \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --eval_steps 1000 \
  --logging_steps 10 \
  --load_best_model_at_end \
  --save_total_limit 1 \
  --metric_for_best_model "accuracy" \
  --early_stopping_patience 6 \
  --report_to "comet_ml"