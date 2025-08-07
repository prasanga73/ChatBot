from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# === MODEL & DATASET CONFIG ===
model_name = "mistralai/Mistral-7B-v0.1"  # or any HF model
dataset_path = "iio_dataset.jsonl"
output_dir = "./qlora-output"

# === LOAD TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# === LOAD MODEL IN 4-BIT FOR QLoRA ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# === PREPARE FOR QLoRA ===
model = prepare_model_for_kbit_training(model)

# === LORA CONFIG ===
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # adjust based on model
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# === LOAD LOCAL DATASET ===
dataset = load_dataset("json", data_files=dataset_path, split="train")

def formatting_func(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"

# === TRAINING ARGS ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    save_total_limit=1,
    fp16=True,
    report_to="none",
    remove_unused_columns=False
)

# === SFT TRAINER ===
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=training_args,
    peft_config=lora_config,
    formatting_func=formatting_func,
)

# === TRAIN MODEL ===
trainer.train()

# === SAVE FINETUNED MODEL ===
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
