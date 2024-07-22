from models import GeminiModel, HFChatModel, HFModel, OpenAIChatModel
from prompt import PromptCreator

# Tasks: ['CodeTAT-QA', 'TAT-QA', 'CodeFinQA', 'FinKnow', 'FinCode', 'ConvFinQA']



cot_prompt_creator = PromptCreator(
    {
        "FinKnow": "prompts/finknow.json",
        "CodeTAT-QA": "prompts/context_cot.json",
        "CodeFinQA": "prompts/codefinqa_cot.json",
        "FinCode": "prompts/fincode_cot.json",
        "ConvFinQA": "prompts/convfinqa.json",
        "TAT-QA": "prompts/tatqa_e.json",
    }
)



hf_token = "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"

_CONFIG = {
    "gemini-pro-cot": lambda: (GeminiModel("gemini-pro"), cot_prompt_creator),

    "gpt-4-cot": lambda: (OpenAIChatModel("gpt-4"), cot_prompt_creator),
    "gpt-3.5-cot": lambda: (OpenAIChatModel("gpt-3.5-turbo"), cot_prompt_creator),
    "ft-gpt-3.5-turbo-0125-cot": lambda: (OpenAIChatModel("ft:gpt-3.5-turbo-0125:personal::9hiX3x3f"), cot_prompt_creator),


    "Mistral-7B-v0.1-cot": lambda: (
        HFModel("mistralai/Mistral-7B-v0.1", generation_kwargs={"max_new_tokens": 256}),
        cot_prompt_creator,
    ),

    "llama-2-7b-chat-cot": lambda: (
        HFChatModel("meta-llama/Llama-2-7b-chat-hf"),
        cot_prompt_creator,
    ),

    "finetuned-llama3-cot": lambda: (
        HFModel("finetuned_llama3_model", generation_kwargs={"max_new_tokens": 256}, hf_token=hf_token),
        cot_prompt_creator,
    ),
    "finetuned-llama3-rag": lambda: (
        HFModel("finetuned_llama3_model", generation_kwargs={"max_new_tokens": 256}, hf_token=hf_token),
        cot_prompt_creator, 
    ),
    
}


def load_config(name):
    return _CONFIG[name]()


def load_hf_config(
    hugging_face_model_name_or_path,
    prompt_style,
    is_chat_model,
    device_map,
    max_new_tokens,
):
    if is_chat_model:
        m = HFChatModel(
            hugging_face_model_name_or_path,
            device_map=device_map,
            generation_kwargs={
                "max_new_tokens": max_new_tokens,
            },
        )
    else:
        m = HFModel(
            hugging_face_model_name_or_path,
            device_map=device_map,
            generation_kwargs={"max_new_tokens": max_new_tokens},
        )
    if prompt_style == "cot":
        p = cot_prompt_creator

    return m, p