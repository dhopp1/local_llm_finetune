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
        self.modelling = import_module("local_llm_finetune.modelling")

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

    def initialize_model(
        self,
        base_model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=0,
        use_rslora=False,
        loftq_config=None,
    ):
        """Initialize an unsloth model to be trained. See unsloth documentation for hyperparameter explanations.
        parameters:
            :base_model_name: str: HuggingFace name of base model to be fine-tuned. Additional options at: https://huggingface.co/unsloth
        """

        (
            self.unsloth_model,
            self.tokenizer,
            self.max_seq_length,
        ) = self.modelling.initialize_model(
            base_model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )

    def format_training_data(
        self,
        prompt_format=None,
        input_split=None,
    ):
        """Format a dataset into the format necessary for unsloth.
        parameters:
            :prompt_format: str: Prompt format of training samples, alpaca by default
            :input_split: str: If you would like to split the "query" column of your dataset into two separate "instruction" and "input" sections, pass a string that separates those two sections. If not passed, the entire "query" string will go into the "instruction" section of the prompt.
        """

        self.train_dataset = self.modelling.format_training_data(
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            prompt_format=prompt_format,
            input_split=input_split,
        )

    def setup_training(
        self,
        dataset_num_proc=2,
        packing=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=0,
        output_dir="outputs",
    ):
        """Setup the unsloth trainer. See unsloth documentation for explanation of parameters."""

        self.trainer = self.modelling.setup_training(
            unsloth_model=self.unsloth_model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            max_seq_length=self.max_seq_length,
            dataset_num_proc=dataset_num_proc,
            packing=packing,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            bf16=bf16,
            logging_steps=logging_steps,
            optim=optim,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
            output_dir=output_dir,
        )

    def train(self):
        "fine-tune the model"
        self.trainer_stats = self.trainer.train()

    def save_model(self, quantization_method="q5_k_m", output_dir="trained_model"):
        """Save the fine-tuned model to disk
        parameters:
            :quantization_method: str: Quantization method for compressing the model to GGUF
            :output_dir: str: Where to save the trained model
        """
        self.unsloth_model.save_pretrained_gguf(
            output_dir, self.tokenizer, quantization_method
        )
