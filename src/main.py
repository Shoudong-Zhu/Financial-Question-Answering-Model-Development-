import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import tqdm

from config import load_config, load_hf_config
from prompt import Output, load_data
from rag import generate_answer, retrieve_documents

if __name__ == "__main__":
    print("Starting script...")  # Debug statement

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prompt_name",
        default=None,
        type=str,
        help="Choose a config model from `config.py` like `Mistral-7B-v0.1-cot`.",
    )
    parser.add_argument(
        "-n",
        "--hugging_face_model_name_or_path",
        default=None,
        type=str,
        help="Name or path to a huggingface model. Do not set if using `model_prompt_name`.",
    )
    parser.add_argument(
        "--prompt_style",
        default="cot",
        type=str,
        help="Choose one of `cot` or `code`. Ignored if using `model_prompt_name`.",
    )
    parser.add_argument(
        "--is_chat_model",
        action="store_true",
        help="Set to true if model is a chat model. Ignored if using `model_prompt_name`.",
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_new_tokens", default=256)
    parser.add_argument("-t", "--answer_parsing_tries_alloted", default=10, type=int)
    args = parser.parse_args()

    print("Arguments parsed...")  # Debug statement

    assert (args.model_prompt_name is None) ^ (
        args.hugging_face_model_name_or_path is None
    )
    if args.model_prompt_name is not None:
        name = args.model_prompt_name
        model, prompt_creator = load_config(name)
        print(f"Loaded config for {name}...")  # Debug statement
    else:
        model, prompt_creator = load_hf_config(
            args.hugging_face_model_name_or_path,
            args.prompt_style,
            args.is_chat_model,
            args.device_map,
            args.max_new_tokens,
        )
        name = args.hugging_face_model_name_or_path.split("/")[-1] + "-" + args.prompt_style
        print(f"Loaded Hugging Face config for {name}...")  # Debug statement

    answer_parsing_tries_alloted = args.answer_parsing_tries_alloted

    data = load_data("benchmark_questions.json")
    print(f"Loaded {len(data)} questions...")  # Debug statement

    os.makedirs("cache", exist_ok=True)
    cache_name = re.findall("(.*)\-[cot|code]", name)[0]
    cache_file = Path(f"cache/{cache_name}.json")

    cache_data = {}
    if cache_file.exists():
        cache_data = json.load(open(cache_file, "r"))
        print("Loaded cache data...")  # Debug statement

    outputs = []
    for question in tqdm.tqdm(data):
        success = False
        tries = 0
        while not success and tries < answer_parsing_tries_alloted:
            messages = prompt_creator.create(question)
            messages_str = json.dumps(messages)
            if messages_str in cache_data:
                completion = cache_data[messages_str]
            else:
                try:
                    # # Use retrieval and generation
                    # retrieved_docs = retrieve_documents(question['query'])
                    # completion = generate_answer(question['query'], retrieved_docs)
                    completion = model(messages)
                    print(f"Model completion: {completion}")  # Debug statement
                    print("-----------------------------------------------")
                    success, output = prompt_creator.parse_output(completion)
                    print(f"success: {success}, output: {output}")  # Debug statement
                except Exception as e:
                    error_message = str(e)
                    if "insufficient_quota" in error_message:
                        print("Insufficient quota, please check your OpenAI plan and billing details.")
                        time.sleep(60)  # Wait for 1 minute before retrying
                    else:
                        print("Model error: ", error_message)
                        completion = ""
                        success = False

            if success:
                cache_data[messages_str] = completion
                with open(cache_file, "w") as cache_obj:
                    json.dump(cache_data, cache_obj)
            else:
                print("Failed to parse...: ")
                print(completion)
            tries += 1
        outputs.append(Output(question.id, output))

    
    # Create the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    with open(f"results/{name}.csv", "w") as output_file:
        output_file.write("\n".join([f"{o.question_id},{o.answer}" for o in outputs]))

    print("Results saved...")  # Debug statement
