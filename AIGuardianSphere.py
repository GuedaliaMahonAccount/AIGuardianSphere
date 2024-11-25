import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    pipeline
)

# Chemins vers les fichiers JSON
hebrew_path = "C:/Users/USER/guedaApp/AIGuardianSphere/hebrew.json"
english_path = "C:/Users/USER/guedaApp/AIGuardianSphere/english.json"

# Charger les fichiers JSON
with open(english_path, "r", encoding="utf-8") as f:
    english_data = json.load(f)

with open(hebrew_path, "r", encoding="utf-8") as f:
    hebrew_data = json.load(f)

# Fusionner les données
all_data = hebrew_data + english_data

# Contexte par défaut
DEFAULT_CONTEXT = (
    "You are not alone. Many people experience fear, anxiety, or trauma in difficult situations. "
    "Take a deep breath. Remember, there are resources and people ready to help you. You can try "
    "relaxation techniques like deep breathing, or visit our resources page for more information. "
    "If you feel overwhelmed, we recommend talking to a professional."
)

# Liens réels
REAL_LINKS = {
    "doctor_link": "https://example.com/find-doctor",
    "video_link": "https://example.com/calm-videos",
    "help_link": "https://example.com/get-help"
}

# Fonction pour corriger les données
def fix_data(data, default_context, default_links):
    """
    Fixes entries in the dataset by ensuring answers exist in their respective context.

    Args:
    - data: List of dictionaries containing 'question', 'context', and 'answer'.
    - default_context: Default context to use if context is empty or missing.
    - default_links: Dictionary of default links to replace placeholders in answers.

    Returns:
    - Fixed dataset where answers are always present in the context.
    """
    fixed_data = []

    for item in data:
        # S'assurer que le contexte n'est pas vide
        if not item.get("context"):
            item["context"] = default_context

        # Remplacer les placeholders par des liens réels
        for placeholder, real_link in default_links.items():
            item["answer"] = item["answer"].replace(f"[{placeholder}]", real_link)
            item["context"] = item["context"].replace(f"[{placeholder}]", real_link)

        # Ajouter la réponse au contexte si elle est absente
        if item["answer"] not in item["context"]:
            item["context"] += f" {item['answer']}"

        fixed_data.append(item)

    return fixed_data

# Corriger les données
all_data = fix_data(all_data, DEFAULT_CONTEXT, REAL_LINKS)

# Valider les données corrigées
for item in all_data:
    if item["answer"] not in item["context"]:
        print(f"Error: Answer not found in context for item: {item}")
    else:
        print(f"Validated: {item['question']}")

# Préparer les données pour Hugging Face Dataset
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

# Préparer le dataset
dataset = prepare_data_for_hf(all_data)

# Diviser les données en ensembles d'entraînement et de validation
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Initialiser le tokenizer et le modèle
model_name = "xlm-roberta-base"
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fonctions de prétraitement
def tokenize_examples(examples):
    return tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

def add_answer_positions(examples):
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

# Appliquer les fonctions de prétraitement
train_dataset = train_dataset.map(tokenize_examples, batched=True)
train_dataset = train_dataset.map(add_answer_positions, batched=True)

eval_dataset = eval_dataset.map(tokenize_examples, batched=True)
eval_dataset = eval_dataset.map(add_answer_positions, batched=True)

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    report_to="none",
)

# Initialiser le formateur
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle et le tokenizer
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

# Tester le modèle
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
