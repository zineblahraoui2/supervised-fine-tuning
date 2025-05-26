import json
import os
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported
from huggingface_hub import login
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import TextStreamer


login(os.environ.get('HUGGING_FACE_TOKEN'))

wandb.login(key=os.environ.get('WANDB_KEY'))
run = wandb.init(
    project='Fine-tune-DeepSeek-R1-Distill-Llama-8B on marketing dataset',
    job_type="training",
    anonymous="allow"
)

print(run.url)

max_seq_length = 2048
dtype = None
load_in_4bit = True


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = token_Z,
)

prompt_style = """Below is an instruction that describes a task, followed by a topic that provides marketing context.
Write a response that generates a persuasive and engaging marketing post.
Before writing, think step by step to ensure the message resonates with the target audience, highlights key value propositions, and includes a clear call-to-action if relevant.

### Instruction:
You are a marketing expert specialized in crafting high-converting, audience-focused content.
Generate a promotional or value-driven post based on the topic below.

### Marketing Topic:
{}

### Response:
<think>{}"""

question = "Launch of our eco-friendly water bottles made from 100% recycled materials. Emphasize sustainability, design, and encourage people to join the green movement."

FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)

response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])

train_prompt_style = """Below is an instruction that describes a task, followed by a topic that provides marketing context.
Write a response that generates a persuasive and engaging marketing post.
Before writing, think step by step to ensure the message resonates with the target audience, highlights key value propositions, and includes a clear call-to-action if relevant.

### Instruction:
You are a marketing expert specialized in crafting high-converting, audience-focused content.
Generate a promotional or value-driven post based on the topic below.

### Marketing Topic:
{}

### Response:
<think>
{}
</think>
{}"""


with open('/content/OurDataset (2).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(articles):
    texts = []
    for article in articles:
        try:
            question = (
                article["input"]["instruction"]
                + " (Audience: " + ", ".join(article["input"]["target_audience"])
                + "; Tone: " + article["input"]["tone"]
                + "; Format: " + article["input"]["format"] + ")"
            )
            cot = (
                "Focus on key concepts like "
                + ", ".join(article["input"]["keywords"])
                + " to address the audience's needs in the "
                + article["metadata"]["sector"] + " sector."
            )
            response = article["output"]["content"]
            full_prompt = train_prompt_style.format(question, cot, response) + EOS_TOKEN
            texts.append({"text": full_prompt})
        except KeyError as e:
            print(f"Skipping an article due to missing field: {e}")
    return texts

formatted_dataset = formatting_prompts_func(data["articles"])



with open("/content/OurDataset (2).json", "r") as f:
    data = json.load(f)

data["articles"]



hf_dataset = Dataset.from_list(data["articles"])

def formatting_prompts_func(batch):
    texts = []
    for i in range(len(batch["input"])):
        input_data = batch["input"][i]
        metadata = batch["metadata"][i]
        output_data = batch["output"][i]

        question = (
            input_data["instruction"]
            + " (Audience: " + ", ".join(input_data["target_audience"])
            + "; Tone: " + input_data["tone"]
            + "; Format: " + input_data["format"] + ")"
        )

        cot = (
            "Focus on key concepts like "
            + ", ".join(input_data["keywords"])
            + " to address the audience's needs in the "
            + metadata["sector"] + " sector."
        )

        response = output_data["content"]
        full_prompt = train_prompt_style.format(question, cot, response) + EOS_TOKEN
        texts.append(full_prompt)

    return {"text": texts}

hf_dataset = hf_dataset.map(formatting_prompts_func, batched=True)

print(hf_dataset["text"][0])

model = FastLanguageModel.get_peft_model(
    model,
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
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)



trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=hf_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()



instruction = "Créer un post LinkedIn pour sensibiliser à l'importance du branding personnel pour les freelances."
audience = ["Freelances", "Consultants", "Indépendants"]
tone = "motivant"
format_ = "post LinkedIn"
keywords = ["branding personnel", "réputation", "marketing", "visibilité"]
sector = "marketing digital"


question = (
    instruction
    + " (Audience: " + ", ".join(audience)
    + "; Tone: " + tone
    + "; Format: " + format_ + ")"
)

cot = (
    "Focus on key concepts like "
    + ", ".join(keywords)
    + " to address the audience's needs in the "
    + sector + " sector."
)


prompt = train_prompt_style.format(question, cot, "")
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")


streamer = TextStreamer(tokenizer)
output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=1500,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    streamer=streamer,
)

prompt = """Tu es un expert marketing.
Rédige un post LinkedIn motivant à destination des freelances, consultants et indépendants.
Thème : L'importance du branding personnel pour réussir dans le marketing digital.
Utilise un ton motivant, inclus les mots-clés : branding personnel, réputation, marketing, visibilité.
Ajoute un appel à l'action à la fin.
"""


inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1500,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
)


response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)

prompt = """Tu es un expert marketing.
Rédige un post LinkedIn motivant pour des freelances, consultants et indépendants sur le sujet du branding personnel dans le marketing digital.
"""

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1500,
    do_sample=True,
    top_p=0.9,
    temperature=1,)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)

