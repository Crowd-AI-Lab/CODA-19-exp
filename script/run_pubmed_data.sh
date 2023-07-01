##############################################################
# We will organize the PubMed data in the desired format
##############################################################

PUBMED_DATA_DIR="data/PubMed_200k_RCT"

echo "Organizing PubMed data without position-encoding"
for split in train dev test; do
    python src/process_data_pubmed.py process-data \
        --input-file "${PUBMED_DATA_DIR}/${split}.txt" \
        --output-file "data/pubmed/${split}.json"
done

echo
echo "Organizing PubMed data with position-encoding"
for split in train dev test; do
    python src/process_data_pubmed.py process-data \
        --input-file "${PUBMED_DATA_DIR}/${split}.txt" \
        --output-file "data/pubmed-position/${split}.json" \
        --position-encoding
done

echo
echo "Mixing CODA-19 and PubMed data with position-encoding"
for split in train dev test; do
    python src/process_data_pubmed.py mix-data \
        --input-file-1 "data/coda19-position/${split}.json" \
        --input-file-2 "data/pubmed-position/${split}.json" \
        --output-file "data/simple-mix-position/${split}.json"
done

echo
echo "Mixing CODA-19 and PubMed data with position-encoding + upsampling"
for split in train dev test; do
    python src/process_data_pubmed.py mix-data \
        --input-file-1 "data/coda19-position/${split}.json" \
        --input-file-2 "data/pubmed-position/${split}.json" \
        --output-file "data/upsampling-mix-position/${split}.json" \
        --upsampling
done

echo
echo "Organize PubMed data using the CODA-19 label set (this is for two-staged fine-tuning)"
for split in train dev test; do
    python src/process_data_pubmed.py process-data \
        --input-file "${PUBMED_DATA_DIR}/${split}.txt" \
        --output-file "data/pubmed-position-coda19-label/${split}.json" \
        --coda19-label \
        --position-encoding
done