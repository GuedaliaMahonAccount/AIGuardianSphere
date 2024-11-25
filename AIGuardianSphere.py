# Step 1: Load JSON Files from Google Drive
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    pipeline
)



# Load Hebrew and English datasets
with open("C:/Users/USER/guedaApp/AIGuardianSphere/english.json", "r", encoding="utf-8") as f:
    english_data = json.load(f)

with open("C:/Users/USER/guedaApp/AIGuardianSphere/hebrew.json", "r", encoding="utf-8") as f:
    hebrew_data = json.load(f)



# Merge datasets into a single list
all_data = hebrew_data + english_data




# Step 2: Add Default Context if Empty
DEFAULT_CONTEXT = (
    "You are not alone. Many people experience fear, anxiety, or trauma in difficult situations. "
    "Take a deep breath. Remember, there are resources and people ready to help you. You can try "
    "relaxation techniques like deep breathing, or visit our resources page for more information. "
    "If you feel overwhelmed, we recommend talking to a professional."
)

def add_default_context(data):
    for item in data:
        if not item["context"]:
            item["context"] = DEFAULT_CONTEXT
    return data

all_data = add_default_context(all_data)






# Step 3: Clean Text in the Dataset
def clean_text(data):
    """
    Cleans the text in the dataset by replacing placeholders and fixing formatting.
    """
    for item in data:
        item["context"] = item["context"].replace("[link]", "this link").replace("[קישור]", "קישור זה")
        item["answer"] = item["answer"].replace("[link]", "this link").replace("[קישור]", "קישור זה")
    return data

all_data = clean_text(all_data)





# Step 4: Prepare Data for Hugging Face Dataset
def prepare_data_for_hf(data):
    """
    Converts data into the format required by the Hugging Face Dataset.
    """
    processed_data = {
        "question": [item["question"] for item in data],
        "context": [item["context"] for item in data],
        "answers": [
            {
                "text": item["answer"],
                "answer_start": [item["context"].index(item["answer"])] if item["answer"] in item["context"] else [0],
            }
            for item in data
        ],
    }
    return Dataset.from_dict(processed_data)

# Prepare dataset
dataset = prepare_data_for_hf(all_data)







# Step 5: Split Data into Training and Evaluation Sets
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]







# Step 6: Initialize Tokenizer and Model
model_name = "deepset/roberta-base-squad2"  # Pre-trained model optimized for QA tasks
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)







# Step 7: Preprocessing Functions
def tokenize_examples(examples):
    return tokenizer(
        examples["question"], examples["context"],
        truncation=True, padding="max_length", max_length=256  # Réduction de la taille max
    )

def add_answer_positions(examples):
    """
    Adds start and end positions for answers within the tokenized context.
    """
    start_positions = []
    end_positions = []
    for i in range(len(examples["answers"])):
        answer = examples["answers"][i]["text"]
        context = examples["context"][i]
        if answer in context:
            start_idx = context.index(answer)
            end_idx = start_idx + len(answer)
        else:
            start_idx = 0
            end_idx = 0
        start_positions.append(start_idx)
        end_positions.append(end_idx)
    examples["start_positions"] = start_positions
    examples["end_positions"] = end_positions
    return examples





# Step 8: Apply Preprocessing Functions to Datasets
train_dataset = train_dataset.map(tokenize_examples, batched=True)
train_dataset = train_dataset.map(add_answer_positions, batched=True)

eval_dataset = eval_dataset.map(tokenize_examples, batched=True)
eval_dataset = eval_dataset.map(add_answer_positions, batched=True)






# Step 9: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,  # Batch plus petit
    gradient_accumulation_steps=1,
    num_train_epochs=3,  # Réduction du nombre d'époques
    weight_decay=0.01,
    save_total_limit=1,
    report_to="none",
)







# Step 10: Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)




# Step 11: Train the Model
trainer.train()






# Step 12: Save the Model and Tokenizer
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")






# Step 13: Test the Trained Model
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)

# Example Test
test_questions = [
    {"question": "I feel scared, what should I do?", "context": ""},
    {"question": "How do I manage my stress?", "context": "I feel overwhelmed after a long shift as a paramedic."},
    {"question": "I'm alone and afraid.", "context": ""}
]

for test in test_questions:
    context = test["context"] if test["context"] else DEFAULT_CONTEXT
    result = qa_pipeline({"question": test["question"], "context": context})
    print(f"Question: {test['question']}")
    print(f"Answer: {result['answer']}")
