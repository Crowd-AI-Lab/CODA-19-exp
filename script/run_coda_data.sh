##############################################################
# We will organize the CODA-19 data in the desired format
##############################################################

CODA19_DATA_DIR="data/CODA19_v1_20200504"

echo "Organizing CODA-19 data without position-encoding"
for split in train dev test; do
    python src/process_data_coda.py \
        --input-folder "${CODA19_DATA_DIR}/human_label/${split}" \
        --output-filename "data/coda19/${split}.json"
done

echo
echo "Organizing CODA-19 data with position-encoding"
for split in train dev test; do
    python src/process_data_coda.py \
        --input-folder "${CODA19_DATA_DIR}/human_label/${split}" \
        --output-filename "data/coda19-position/${split}.json" \
        --position-encoding
done