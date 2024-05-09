import os

from local_rag_llm.model_setup import instantiate_llm
from local_rag_llm.local_llm import local_llm
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
import torch
from datasets import Dataset

from helper.data_prep import clean_up_response, process_data

if False:
    llm = instantiate_llm(
        llm_url="",
        llm_path="models/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
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
    
    metadata = pd.read_csv("data/tdr/metadata.csv") # with metadata + column named "filepath" for location of .txt files
    
    # llm responses
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
    dataset = dataset.loc[lambda x: ~pd.isna(x.response),:].loc[lambda x: ~x.response.str.startswith(" I'd be happy"), :].reset_index(drop=True)
    
    # write out dataset
    dataset.to_csv("data/tdr_dataset.csv", index=False)
    
    # text completion
    dataset = process_data(
        metadata=metadata,
        llm=None,
        model=None,
        chunk_size=1024,
        overlap_size=200,
        prompt_format="This is an excerpt from the document with the the following metadata: {}. It is currently about 1000 words long. Complete the excerpt with a new text about one third as long. Here is the excerpt: '{}'",
        completion_ratio=0.75,
    )
    
    # write out dataset
    dataset.to_csv("data/tdr_completion_dataset.csv", index=False)

# unsloth training
dataset = pd.read_csv("data/tdr_dataset.csv").iloc[:3,:]

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = os.environ['HF_TOKEN'],
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# prompt format
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

# convert data into final format
train_data = [{"output": dataset.loc[i, "response"], "input": dataset.loc[i, "query"].split(" Here is the excerpt: ")[1], "instruction": dataset.loc[i, "query"].split(" Here is the excerpt: ")[0]} for i in range(len(dataset))]
train_dataset = Dataset.from_list(train_data)
train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)

# setup training
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# train the model
trainer_stats = trainer.train()

# save the model
model.save_pretrained_gguf("finetuned_models/tdr", tokenizer, quantization_method = "q5_k_m")