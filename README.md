# CODA-19-exp
This repo contains code for the paper, Good Data, Large Data, or No Data? Comparing Three Approaches in Developing Research Aspect Classifiers for Biomedical Papers.

We are organizing the code. Stay tuned!


## Getting Data
In this project, we conducted experiments using [CODA-19](https://github.com/windx0303/CODA-19) and [PubMed](https://github.com/Franck-Dernoncourt/pubmed-rct) dataset.
Please download both of the data from the corresponding Github repo.

### CODA-19
Once you download the data from https://github.com/windx0303/CODA-19.
Unzip the `data/CODA19_v1_20200504.zip` from the repo and move the `CODA19_v1_20200504` folder to our `data` folder.

You can run `script/run_coda_data.sh` directly to process all the data in batch. The script will generate (1) coda19 and (2) coda19-position data in a json format under the `data` folder.
The detailed step-by-step settings are described below.
```bash
sh script/run_coda_data.sh
```

#### **Details about the Command**
To process the CODA19 data in the json format, run the following command. Note that the command will only process the `train` data. Please modify the command accordingly for `dev` and `test` sets.
```bash
python src/process_data_coda.py \
    --input-folder "data/CODA19_v1_20200504/human_label/train" \
    --output-filename "data/coda19/train.json"
```

You can enable the position-encoding by adding the `position-encoding` flag.
```bash
python src/process_data_coda.py \
    --input-folder "data/CODA19_v1_20200504/human_label/train" \
    --output-filename "data/coda19-position/train.json" \
    --position-encoding
```

### PubMed
Once you download the data from https://github.com/Franck-Dernoncourt/pubmed-rct. We are using the `PubMed_200k_RCT` data. Please unzip the `PubMed_200k_RCT/train.7z` and move the `PubMed_200k_RCT` under the `data` folder.

You can run `script/run_pubmed_data.sh` directly to process all the data in batch. This script will generate (1) pubmed, (2) pubmed-position, (3) simple-mix-position, (4) upsampling-mix-position, (5) pubmed-position-coda19-label under the `data` folder.
The detailed step-by-step settings are described below.
```bash
sh script/run_pubmed_data.sh
```

#### **Details about the Command**
To process the PubMed data in the json format, run `process-data` in `src/process_data_pubmed.py`. Note that the command will only process the `train` data. Please modify the command accordingly for `dev` and `test` sets.
```bash
python src/process_data_pubmed.py process-data \
        --input-file "data/PubMed_200k_RCT/train.txt" \
        --output-file "data/pubmed/train.json"
```

Again, you can add the position-encoding to the text by specifying the `--position-encoding` flag.
```bash
python src/process_data_pubmed.py process-data \
        --input-file "data/PubMed_200k_RCT/train.txt" \
        --output-file "data/pubmed/train.json" \
        --position-encoding
```

To generate the PubMed data using the CODA-19 label set, specify the `--coda19-label` flag. Note that this is for the `two-staged training` experiment.
```bash
python src/process_data_pubmed.py process-data \
        --input-file "data/PubMed_200k_RCT/train.txt" \
        --output-file "data/pubmed/train.json" \
        --position-encoding \
        --coda19-label
```

To generate the mixed data (CODA19+PubMed) for the `simple-mixing` and `upsampling-mixing` experiment, please use `mix-data` in `src/process_data_pubmed.py`. Note that we are mixing the position-encoded data and the order of `input-file-1` and `input-file-2` does not matter.
```bash
python src/process_data_pubmed.py mix-data \
        --input-file-1 "data/coda19-position/train.json" \
        --input-file-2 "data/pubmed-position/train.json" \
        --output-file "data/simple-mix-position/train.json"
```

To upsample the data with fewer samples, specify the `--upsampling` flag. You can also change the `ratio` (default value is 10) to determine the upsampling ratio.
```
python src/process_data_pubmed.py mix-data \
        --input-file-1 "data/coda19-position/train.json" \
        --input-file-2 "data/pubmed-position/train.json" \
        --output-file "data/simple-mix-position/train.json" \
        --upsampling \
        [--ratio 10]
```

## Fine-tuned Model Experiment



## LLM Experiment

