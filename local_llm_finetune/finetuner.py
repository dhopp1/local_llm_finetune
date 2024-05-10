from importlib import import_module
import os

from local_rag_llm.model_setup import instantiate_llm
from local_rag_llm.local_llm import local_llm
import torch
import pandas as pd


class finetuner:
    """Primary class of the library, setup and run the finetune
    parameters:
        :metadata_path: str: optional, location of the metadata CSV file if setting up a new training dataset. It can have any desired columns, but must have at least one column named "filepath" with the location of the .txt files for training data
        :llm_path: str: location of the LLM if setting up training data via responses from an existing LLM
        :llm_gpu_layers: int: number of LLM layers to offload to the GPU
        :llm_context_window: int: size of the LLM context window
        :llm_temperature: float: number between 0 and 1, 0 = more conservative/less creative, 1 = more random/creative
        :llm_max_new_tokens: int: limit of how many tokens to produce for an answer
        :chunk_size: int: when creating a synthetic training set, will take each full document and split it into chunks of this many tokens
        :chunk_overlap: int: how much overlap to have for each chunk
        :prompt_format: str: query for the LLM to produce synthetic responses, or for text completion. Should have two {}'s, the first one will be filled in with the metadata, the second one with the chunk's content
        :completion_ratio: float: if not using an LLM to create synthetic responses, what proportion of the chunk to give in the query
        :cleanup_model_family: str: "llama-3" for Llama 3. It will cleanup LLM responses for unhelpful/extraneous content
        :dataset: pd.DataFrame: optional, an already prepared dataset. Must contain two columns named "query" and "response"
    """

    def __init__(
        self,
        metadata_path=None,
        llm_path=None,
        llm_gpu_layers=100,
        llm_context_window=4000,
        llm_temperature=0.0,
        llm_max_new_tokens=512,
        chunk_size=1024,
        chunk_overlap=200,
        prompt_format="This is an excerpt from the document with the the following metadata: {}. It is currently about 750 words long. Complete the excerpt with a new text about one third as long. Here is the excerpt: '{}'",
        completion_ratio=0.75,
        cleanup_model_family=None,
        dataset=None,
    ):
        self.data_prep = import_module("local_llm_finetune.data_prep")

        self.metadata_path = metadata_path
        self.llm_path = llm_path
        self.llm_gpu_layers = llm_gpu_layers
        self.llm_context_window = llm_context_window
        self.llm_temperature = llm_temperature
        self.llm_max_new_tokens = llm_max_new_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.prompt_format = prompt_format
        self.completion_ratio = completion_ratio
        self.cleanup_model_family = cleanup_model_family
        self.dataset = dataset

        # initializing metadata
        if self.metadata_path != None:
            self.metadata = pd.read_csv(self.metadata_path)
        else:
            self.metadata = None

        # initializing LLM
        if self.llm_path != None:
            self.llm = instantiate_llm(
                llm_url="",
                llm_path=self.llm_path,
                redownload_llm=False,
                n_gpu_layers=self.llm_gpu_layers,
                context_window=self.llm_context_window,
            )

            self.model = local_llm(
                text_path=None,
                temperature=self.llm_temperature,
                max_new_tokens=self.llm_max_new_tokens,
            )
        else:
            self.llm = None
            self.model = None

    def prepare_dataset(self, dataset_out_path=None):
        """Create a fine-tuning dataset from raw documents
        parameters:
            :dataset_out_path: str: where to save the dataset
        """
        self.dataset = self.data_prep.process_data(
            metadata=self.metadata,
            llm=self.llm,
            model=self.model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            prompt_format=self.prompt_format,
            completion_ratio=self.completion_ratio,
        )

        # if using LLM responses, clean up unhelpful outputs
        if self.cleanup_model_family != None:
            self.dataset["response"] = [
                self.data_prep.clean_up_response(x, self.cleanup_model_family)
                for x in self.dataset["response"]
            ]

            if self.cleanup_model_family == "llama-3":
                self.dataset = (
                    self.dataset.loc[lambda x: ~pd.isna(x.response), :]
                    .loc[lambda x: ~x.response.str.startswith(" I'd be happy"), :]
                    .reset_index(drop=True)
                )

        # write the dataset out
        if dataset_out_path != None:
            self.dataset.to_csv(dataset_out_path, index=False)
