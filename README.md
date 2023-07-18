# CODA-19-exp
This repo contains code for the paper, Good Data, Large Data, or No Data? Comparing Three Approaches in Developing Research Aspect Classifiers for Biomedical Papers.

You can find the paper here: https://aclanthology.org/2023.bionlp-1.8/

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

### Inference and Evaluate
After training the model, you can use `src/predict.py` to make prediction for the given data.
1. `model-name-or-path`: the path for the fine-tuned model (e.g., `model/coda19-position-scibert_scivocab_uncased`)
2. `test-filename`: the path for the testing data
3. `text-key`: the key (json field name) for the input text in each of the sample (default: text)
4. `output-filename`: the path for the output result
5. `batch-size`: number of samples per batch (default: 32)
6. `device`: device used for prediction (default: cuda:0)
7. `label-mapping`: whether turn the PubMed label into CODA19 label set (default: False)

The following is an example command.
```bash
python src/predict.py \
        --model-name-or-path model/coda19-position-scibert_scivocab_uncased \
        --test-filename data/coda19-position/test.json \
        --output-filename output/prediction-coda19-position-scibert.txt
```

After getting the predictions, you can run `src/compute score.py` to compute the score. There are only two arguments.
1. `predict-file`: the path for the prediction file (e.g., output/prediction-coda19-position-scibert.txt)
2. `answer-file`: the path for the test file

```bash
python src/compute_score.py \
        --predict-file output/prediction-coda19-position-scibert.txt \
        --answer-file data/coda19-position/test.json
```
You will see the output like this.
```
              precision    recall  f1-score   support

  background   0.824990  0.794350  0.809380      5062
     finding   0.823230  0.867199  0.844642      6890
      method   0.741281  0.665421  0.701305      2140
       other   0.818653  0.843416  0.830850       562
     purpose   0.638197  0.655298  0.646635       821

    accuracy                       0.803360     15475
   macro avg   0.769270  0.765137  0.766562     15475
weighted avg   0.802490  0.803360  0.802280     15475

Micro F1 Score 0.8033602584814217
```

## LLM Experiment
In this paper, we run 6 large language models (LLMs), including 3 open-sourced LLMs (LLaMA, MPT, and Dolly) and 3 closed LLMs (GPT3, ChatGPT, and GPT4).
We test them in both `zero-shot` and `few-shot` settings. You can find the **prompts** in the `prompt` folder (`prompt/zero-shot.txt` and `prompt/few-shot.txt`).

The subset of the samples (1250 samples) we used for our experiment is included in the `data` folder.

### Preparing the models
**Open-AI models**

Please set your OpenAI API key as an environment variable (`OPENAI_API_KEY`)
```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
If you are using conda, you can also use conda to manage it. Remember to re-activate your conda environment!
```bash
conda env config vars set OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**LLaMA**

We are using the officially released LLaMA model. Please fill out the request form and obtain the model [here](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform).

You can then convert the LLaMA weights into HuggingFace's format following the instruction [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py).

### Prediction
The interface for generating responses is written in `src/llm.py`. Here, to run the prediction, we will be calling `src/llm_exp.py`.
1. `model-type`: the model type you would like to use (e.g., gpt3, chatgpt, gpt4, llama, mpt, dolly)
2. `model-path`: if you are using the open-sourced models (llama, mpt, dolly), you will need to specify the precise model or path to the model (e.g., `databricks/dolly-v2-12b` or `/data/data/llama/llama-7B`)
3. `test-file`: the path for the test set.
4. `prompt-file`: the path for the prompt file.
5. `output-folder`: the folder path for all the output files.
Note that if there are 20 samples in the test file, then there will be 20 files in the `output-folder` after execution.

Here is an example command we used to run LLaMA with the zero-shot setting.
```bash
python src/llm_exp.py exp \
        --model-path /data/data/share/llama/llama-65B \
        --model-type llama \
        --test-file data/test_subset.json \
        --prompt-file prompt/zero-shot.txt \
        --output-folder output/llama-zero-shot
```

Running GPT4 with the few-shot setting.
```bash
python src/llm_exp.py exp \
        --model-type gpt4
        --test-file data/test_subset.json \
        --prompt-file prompt/few-shot.txt \
        --output-folder output/gpt4-few-shot
```

### Calculate Scores
We implement the `compute-score` command in `src/llm_exp.py`. To use it, you will need to specify the following arguments:
1. `test-file`: the path for the test file
2. `output-folder`: the path for the LLM's output folder (e.g., output/gpt4-few-shot)
3. `model-type`: the model type for this LLM. We need this to determine how to extract the generated text from the responses.

Here is an example command for computing scores for the GPT-4 + few-shot output.
```bash
python src/llm_exp.py compute-score \
        --model-type gpt4 \
        --test-file data/test_subset.json \
        --output-folder output/gpt4-few-shot
```

### Batch Processing
If you would like to run all the experiment, you can use the shell scripts we provided in the `script` folder.

**Inference in Batch**
```bash
bash script/run_llm_exp.sh
```

**Compute Scores in Batch**
```bash
bash script/compute_llm_scores.sh
```

## Citation
If you use the code or results from the paper, please consider citing our paper.
```bibtex
@inproceedings{chandrasekhar-etal-2023-good,
    title = "Good Data, Large Data, or No Data? Comparing Three Approaches in Developing Research Aspect Classifiers for Biomedical Papers",
    author = "Chandrasekhar, Shreya  and
      Huang, Chieh-Yang  and
      Huang, Ting-Hao",
    booktitle = "The 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks",
    month = july,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.bionlp-1.8",
    pages = "103--113",
}
```
