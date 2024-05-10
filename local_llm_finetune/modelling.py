import os

from datasets import Dataset
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel


def initialize_model(
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
    "initialize an unsloth base model to be finetuned"

    unsloth_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=os.environ["HF_TOKEN"],
    )

    unsloth_model = FastLanguageModel.get_peft_model(
        unsloth_model,
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

    return unsloth_model, tokenizer, max_seq_length


def format_training_data(
    tokenizer,
    dataset,
    prompt_format=None,
    input_split=None,
):
    "format training data into format for unsloth"
    if prompt_format == None:
        prompt_format = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = prompt_format.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    pass

    # convert data into final format
    if input_split != None:
        train_data = [
            {
                "output": dataset.loc[i, "response"],
                "input": dataset.loc[i, "query"].split(input_split)[1],
                "instruction": dataset.loc[i, "query"].split(" Here is the excerpt: ")[
                    0
                ],
            }
            for i in range(len(dataset))
        ]
    else:
        train_data = [
            {
                "output": dataset.loc[i, "response"],
                "input": "",
                "instruction": dataset.loc[i, "query"],
            }
            for i in range(len(dataset))
        ]

    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    return train_dataset


def setup_training(
    unsloth_model,
    tokenizer,
    train_dataset,
    max_seq_length=2048,
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
    "setup the trainer"

    trainer = SFTTrainer(
        model=unsloth_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=dataset_num_proc,
        packing=packing,
        args=TrainingArguments(
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
        ),
    )

    return trainer


def train(trainer):
    "fine-tune the model"

    trainer_stats = trainer.train()

    return trainer_stats


def save_model(
    unsloth_model, tokenizer, quantization_method="q5_k_m", output_dir="trained_model"
):
    "save the fine-tuned model to disk"
    unsloth_model.save_pretrained_gguf(output_dir, tokenizer, quantization_method)
