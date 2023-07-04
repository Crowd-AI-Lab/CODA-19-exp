from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM, 
    AutoConfig,
    GenerationConfig, 
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import transformers
import torch
import typer
import json
from tqdm import tqdm
from pathlib import Path
import openai
from typing import Dict, List
import os
import re

class GPT3():
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.space_pattern = re.compile(r"\s+")
        self.parameters = {
            "model": "text-davinci-003",
            "temperature": 0.0,
            "max_tokens": 20,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    def generate(self, prompt:str) -> Dict:
        response = openai.Completion.create(
            prompt=prompt,
            **self.parameters,
        )
        return response

class ChatGPT():
    def __init__(self, model: str = "gpt4"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.space_pattern = re.compile(r"\s+")
        self.parameters = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 20,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    def generate(self, prompt:List) -> Dict:
        response = openai.ChatCompletion.create(
            messages=prompt,
            **self.parameters,
        )
        return response


class LLM:
    def __init__(
        self,
        model_type: str,
        model_name_or_path: str,
    ):
        self.model_type = model_type

        if model_type == "llama":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
            self.model = LlamaForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto", 
            )
            self.model.eval()
        elif model_type == "gpt2":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto", 
            )
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
            self.model.eval()
        else:
            # for mpt and dolly
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=self.config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model.eval()
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # generation config
        self.generation_kwargs = GenerationConfig(**{
            "top_p": 0.95, 
            "do_sample": True,
            "eos_token_id": self.model.config.eos_token_id,
            "pad_token_id": self.model.config.pad_token_id,
            "min_new_tokens": 10,
            "max_new_tokens": 30,
            "repetition_penalty": 1.0,
            "temperature": 0.1,
            "num_beams": 1,
        })

    @torch.no_grad()
    def generate(
        self,
        text: str,
    ) -> Dict:
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # move to cuda
        inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_kwargs,
        )
        return {
            "text": self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True),
            "response": self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        }
