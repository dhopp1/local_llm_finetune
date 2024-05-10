# local_llm_finetune
Fine-tune an LLM with Unsloth.

# Installation
- Download the `docker-compose.yml` and `Dockerfile`
- Edit the `HF_TOKEN` in `docker-compose.yml` to your API token
- Under `volumes:` in `docker-compose.yml`, map local directories to that in the container as necessary. These directories should contain the script you want to run, any metadata files/training data, LLMs if necessary, etc.
- Navigate to the directory you saved the .yml file and run `docker compose up`
- Check the name of the image with `docker ps -a`
- Run your desired local script with `docker exec -t <image name from previous step> python /app/<your script>.py`, being sure to use the container's directory structure in the script.

# Usage
## Create training data
The library uses a Pandas DataFrame as the basis of the training data. This dataframe should have two columns, `query`, and `response`, corresponding to what you are asking the model and what its response should be. If you don't have your data set up in this way and only have raw documents, you can use the library to get it into this format.

First, convert your documents into `.txt` format. Then, create a `metadata.csv` file. This file can contain any columns desired, but must at least have one column named `filepath`. This column should contain the location of the `.txt` files. Make sure it uses the path structure of the container, not your local machine.

There are two methods for converting the documents to query-response pairs:
	1. Synthetic LLM responses
	2. Text completion responses

In both cases the documents will be split into chunks as long as specified in the `chunk_size` parameter. 

The first method will query an existing LLM with the prompt format specified in the `prompt_format` parameter. This string should be in a format like `"Metadata: {}, instruction: {}"`. Where the first brackets will be filled in with the document's metadata, and the second with the chunk's content. This prompt will become the `query` column in the eventual `dataset` output. The LLM's response will become the `response` column.

The second method will split the chunk into two sections, with the ratio determined by the `completion_ratio` parameter. E.g., if the parameter is `0.75`, 75% of the chunk will go into the prompt, and the remaining 25% will go in as the response. Below is a code example of both methods.

```py
from local_llm_finetune.finetuner import finetuner

# LLM response method
finetune = finetuner(
    metadata_path="container_path_to_metadata/metadata.csv",
    llm_path="container_path_to_GGUF_llm/model.gguf", # location of LLM, if not passed, will default to text completion method
    chunk_size=1024,
    chunk_overlap=200,
    prompt_format="This is an excerpt from the document with the the following metadata: {}. It is currently about 1000 words long. Summarize the information to about 100 words, keeping special note of key figures, statistics, and policy recommendations made. Here is the excerpt: '{}'",
)

# prepare the dataset
finetune.prepare_dataset("container_path_out/dataset.csv")

# Text completion method
finetune = finetuner(
    metadata_path="container_path_to_metadata/metadata.csv",
    prompt_format="This is an excerpt from the document with the the following metadata: {}. It is currently about 750 words long. Complete the excerpt with a new text about one third as long. Here is the excerpt: '{}'",
)

# prepare the dataset
finetune.prepare_dataset("container_path_out/dataset.csv")
```

## Fine-tuning a model
```py
import pandas as pd
from local_llm_finetune.finetuner import finetuner

# assuming you created a dataset.csv file in the previous step. If you're in the same script as from above, finetuner.dataset will be created via the finetuner.prepare_dataset function, no need to reinstantiate
dataset = pd.read_csv("dataset.csv")

finetune = finetuner(
    dataset = dataset,
)

# initialize the base model to be fine-tuned
finetune.initialize_model(
    base_model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
)

# format the training data
finetune.format_training_data(
    input_split = "Here is the excerpt:", # how to split the 'query' string of the data, in case you want the first part to go into the 'instruction' section of the alpaca prompt, and the second part to go into the 'input' section. If not passed, all the text in the 'query' column will go into the 'instruction' section
)

# setup the trainer
finetune.setup_training(
    output_dir="full_model", # where to save the full, unquantized model
)

# train the model
finetune.train()

# save the model to GGUF
finetune.save_model(
    output_dir="gguf_model", # where to save the GGUF
)
```