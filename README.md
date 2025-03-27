# local_llm_finetune
Fine-tune an LLM with Unsloth.

# Installation
- install unsloth with:

```
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install unsloth
```
- install the library with `pip install local-llm-finetune`


# Usage
## Create training data
The library uses a Pandas DataFrame as the basis of the training data. This dataframe should have two columns, `query`, and `response`, corresponding to what you are asking the model and what its response should be. If you don't have your data set up in this way and only have raw documents, you can use the library to get it into this format.

First, convert your documents into `.txt` format. Then, create a `metadata.csv` file. This file can contain any columns desired, but must at least have one column named `filepath`. This column should contain the filenames of the `.txt` files.

There are two methods for converting the documents to query-response pairs:
	1. Synthetic LLM responses
	2. Text completion responses

In both cases the documents will be split into chunks as long as specified in the `chunk_size` parameter. 

The first method will query an existing LLM with the prompt format specified in the `prompt_format` parameter. This string should be in a format like `"Metadata: {}, instruction: {}"`. Where the first brackets will be filled in with the document's metadata, and the second with the chunk's content. This prompt will become the `query` column in the eventual `dataset` output. The LLM's response will become the `response` column.

The second method will split the chunk into two sections, with the ratio determined by the `completion_ratio` parameter. E.g., if the parameter is `0.75`, 75% of the chunk will go into the prompt, and the remaining 25% will go in as the response. Below is a code example of both methods.

```py
from local_llm_finetune.data_prep import process_data
import pandas as pd

metadata = pd.read_csv("path/metadata.csv")
files_path = "path_to_txt_files/"

# LLM dataset
dataset = process_data(
    metadata=metadata,
    files_path=files_path,
    chunk_size=500,
    chunk_overlap=150,
    prompt_format="This is an excerpt from the document with the the following metadata: {}. It is currently about 500 words long. Summarize the information to about 100 words. Here is the excerpt:\n\n'{}'",
    llm_url=llm_url, # this should be the url of the LLM to use, for instance, 'http://localhost:8081/v1/' for a llama.cpp local LLM, or 'https://generativelanguage.googleapis.com/v1beta/openai/' for google
    llm_model=model_name, # can be anything for a local LLM, for a cloud provider, the name of the model, e.g., 'gemini-2.0-flash'
    api_Key=api_key, # api key for the cloud provider, 'sk-no-key-required' for a local llama.cpp server
)

# text completion dataset
dataset = process_data(
    metadata=metadata,
    files_path=files_path,
    chunk_size=1000,
    chunk_overlap=200,
    prompt_format="This is an excerpt from the document with the the following metadata: {}. It is currently about 750 words long. Complete the excerpt with a new text about one third as long. Here is the excerpt: '{}'",
    completion_ratio=0.75, # what % of each chunk to have as the question, 1-ratio = what proportion to have as the answer
)

```

## Fine-tuning a model
```py
from local_llm_finetune.modelling import format_training_data, initialize_model, save_model, setup_training, train

# set up the model training
unsloth_model, tokenizer, max_seq_length = initialize_model(
    base_model_name="unsloth/Llama-3.1-8B-Instruct-bnb-4bit", # set to whatever base model to finetune on HuggingFace
    max_seq_length=2048,
)

# train the model
train_dataset = format_training_data(
    tokenizer,
    dataset,
    input_split=" Here is the excerpt:", # how to split the question into 'Instruction' and 'Input'
)

trainer = setup_training(
    unsloth_model,
    tokenizer,
    train_dataset,
    max_seq_length=2048,
    output_dir="outputs",
)

trainer_stats = train(trainer)

# save the output to GGUF
save_model(
    unsloth_model, tokenizer, quantization_method="q5_k_m", output_dir="trained_model"
)
```