import logging
import os
from dataclasses import dataclass

import genai
import torch
from openai import OpenAI
from retry import retry
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class Model:
    def __call__(self, data) -> str:
        ...


@dataclass
class OpenAIChatModel(Model):
    def __init__(self, model, model_kwargs=None):
        self.client = OpenAI(api_key="your openai api key")
        self.model = model

        self.model_kwargs = model_kwargs
        if self.model_kwargs is None:
            self.model_kwargs = {}

    @retry(delay=1, logger=logger, tries=5)
    def __call__(self, messages) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.model_kwargs,
        )

        return completion.choices[0].message.content


class HFModel(Model):
    def __init__(
        self,
        model_name_or_path,
        device_map=0,
        generate_until=None,
        model_kwargs=None,
        generation_kwargs=None,
        hf_token=None,  # Add this parameter
    ):
        self.model_kwargs = model_kwargs
        if self.model_kwargs is None:
            self.model_kwargs = {}

        self.generation_kwargs = generation_kwargs
        if self.generation_kwargs is None:
            self.generation_kwargs = {}

        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map=device_map, use_auth_token=hf_token, **self.model_kwargs  # Add use_auth_token
            )
        else:
            # This will not work for larger models.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, use_auth_token=hf_token, **self.model_kwargs  # Add use_auth_token
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=hf_token)  # Add use_auth_token
        self.model.config.pad_token_id = (
            self.model.config.eos_token_id
        ) = self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.stop_token = None
        if generate_until:
            tokens = self.tokenizer.encode("\n" + generate_until)
            stop_token = list(
                filter(lambda t: self.tokenizer.decode(t) == generate_until, tokens)
            )

            assert len(stop_token) == 1, "Can't parse tokenizer output!"
            self.stop_token = stop_token[0]

class HFChatModel(HFModel):
    def _render(self, messages):
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        return encodeds


def transform_to_gemini(messages_chatgpt):
    messages_gemini = []
    system_promt = ""
    for message in messages_chatgpt:
        if message["role"] == "system":
            system_promt = message["content"]
        elif message["role"] == "user":
            messages_gemini.append({"role": "user", "parts": [message["content"]]})
        elif message["role"] == "assistant":
            messages_gemini.append({"role": "model", "parts": [message["content"]]})
    if system_promt:
        messages_gemini[0]["parts"].insert(0, f"*{system_promt}*")

    return messages_gemini


@dataclass
class GeminiModel(Model):
    def __init__(self, model_name):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    @retry(delay=1, logger=logger, tries=5)
    def __call__(self, messages) -> str:
        response = self.model.generate_content(transform_to_gemini(messages))
        return response.text