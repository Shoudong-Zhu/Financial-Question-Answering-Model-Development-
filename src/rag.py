from datasets import load_dataset

# Load the dataset
dataset = load_dataset("llmware/rag_instruct_benchmark_tester")

# Explore the dataset
print(dataset["train"].features)

# Prepare data
queries = [example["query"] for example in dataset["train"]]
contexts = [example["context"] for example in dataset["train"]]
answers = [example["answer"] for example in dataset["train"]]
categories = [example["category"] for example in dataset["train"]]



from sentence_transformers import SentenceTransformer, util

# Load a pre-trained model for embeddings
retriever_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Encode the contexts
context_embeddings = retriever_model.encode(contexts, convert_to_tensor=True)

def retrieve_documents(query, top_k=5):
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, context_embeddings, top_k=top_k)
    retrieved_docs = [contexts[hit['corpus_id']] for hit in hits[0]]
    return retrieved_docs


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_path = "finetuned_llama3_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def generate_answer(query, retrieved_docs):
    input_text = query + " ".join(retrieved_docs)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
