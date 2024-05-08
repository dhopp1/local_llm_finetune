from local_rag_llm.model_setup import instantiate_llm
from local_rag_llm.local_llm import local_llm
import pandas as pd

from helper.data_prep import clean_up_response, process_data

llm = instantiate_llm(
    llm_url="",
    llm_path="/Users/danielhopp/dhopp1/streamlit_rag/models/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
    redownload_llm=False,
    n_gpu_layers=100,
    context_window=8000,
)

model = local_llm(
    text_path=None,
    temperature=0.0,
    max_new_tokens=512,
)

# TDR example
metadata = pd.read_csv("data/tdr/metadata.csv")

dataset = process_data(
    metadata=metadata,
    llm=llm,
    model=model,
    chunk_size=512,
    overlap_size=100,
    prompt_format="This is an excerpt from the document with the the following metadata: {}. It is currently about 500 words long. Summarize the information to about 100 words, keeping special note of key figures, statistics, and policy recommendations made. Here is the excerpt: '{}'",
)

# clean up of responses
dataset["response"] = [clean_up_response(x, "llama-3") for x in dataset["response"]]

# write out dataset
dataset.to_csv("data/tdr_dataset.csv", index=False)
