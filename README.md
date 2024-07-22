# Fine-Tuning LLMs for Financial Question Answering

## Overview
This project involves the development and fine-tuning of large language models (LLMs) for retrieval-augmented generation (RAG) in the financial domain. The goal is to enhance the performance of models in answering financial questions accurately and reliably, making them suitable for use by analysts and industry professionals.

## Features
- **Fine-Tuning Models**: Implemented fine-tuning of GPT-3.5-turbo and Llama-3 models using PEFT and LoRa techniques.
- **Retrieval-Augmented Generation**: Integrated RAG techniques with the RAG-Instruct-Benchmark-Tester dataset to improve model accuracy in financial question answering.
- **Model Validation**: Validated and benchmarked model performance using a custom test dataset, ensuring high precision and reliability.

## Datasets
- **FinanceBench**: Used for supervised fine-tuning, consisting of 150 annotated examples of financial questions and answers.
- **RAG-Instruct-Benchmark-Tester**: Employed for retrieval-augmented generation, containing 200 samples of financial queries with context passages.

## Project Structure
|-- finetuned_llama3_model/
| |-- adapter_config.json
| |-- adapter_model.safetensors
| |-- README.md
| |-- special_tokens_map.json
| |-- tokenizer_config.json
| |-- tokenizer.json
|-- src/
| |-- config.py
| |-- main.py
| |-- models.py
| |-- prompt.py
| |-- python.py
| |-- chainlit_app.py
|-- benchmark_questions.json
|-- requirements.txt
|-- README.md


## Getting Started

### Prerequisites
- Python 3.7 or higher
- Hugging Face Transformers
- OpenAI API Key
- PyTorch
- Datasets (FinanceBench, RAG-Instruct-Benchmark-Tester)

### Installation
Install the required packages:
pip install -r requirements.txt
