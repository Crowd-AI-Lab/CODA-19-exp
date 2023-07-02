# CODA-19-exp
This repo contains code for the paper, Good Data, Large Data, or No Data? Comparing Three Approaches in Developing Research Aspect Classifiers for Biomedical Papers.

We are organizing the code. Stay tuned!

## Environment Setup
If you use conda to manage your environment, you can create a new one using our `environment.yml` file. If you plan to use `comet_ml` to monitor the training status, you will need to modify the `COMET_API_KEY`. If you plan to run chatGPT or GPT-4 experiment, you will need to modify the `OPENAI_API_KEY`. Both of them can be specified in the `environment.yml` file.

**environment.yml**
```yml
variables:
  COMET_API_KEY: ooooooooooooooooooooooooooo
  OPENAI_API_KEY: sk-oooooooooooooooooooooooooooooooooooooooo
```

To create a new conda environment named `coda19-exp`, please run the following command.
```bash
conda env create --name coda19-exp --file environment.yml
conda activate conda19-exp
```

If you have your own python installed, you can also use `pip` to install all the packages.
```bash
python -m pip install requirements.txt
```


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
We modify the `src/run_glue.py` file provided by HuggingFace to fine-tune all the models.
All the basic settings are included in the `script/train_arg.sh` for calling `src/run_glue.py`.
You will need to specify `data_folder`, `output_folder`, `model_name`, and `cuda_device` when running `script/train_arg.sh`. For example, the following command will fine-tune `allenai/scibert_scivocab_uncased` with `data/coda19-position` dataset using `cuda:0`. The output model will be saved to `model/coda19-position-scibert_scivocab_uncased` folder.
```bash
sh script/train_arg.sh \
  "data/coda19-position" \
  "model/coda19-position-scibert_scivocab_uncased" \
  "allenai/scibert_scivocab_uncased" \
  0
```

In this paper, we fine-tuned a total of 7 models:
1. coda19
2. coda19-position
3. pubmed
4. pubmed-position
5. simple-mix-position
6. upsampling-mix-position
7. two-staged-mix-position: train the model with `pubmed-position-coda19-label` and `coda19-position` sequentially

To run all the experiment at once, you can run `script/train_exp.sh` (the default gpu is cuda:0, please modify it accordingly).
```bash
sh script/train_exp.sh
```

## LLM Experiment


