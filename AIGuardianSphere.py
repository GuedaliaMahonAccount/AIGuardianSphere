import os
from transformers import pipeline

# Désactiver les messages TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Charger un pipeline de questions-réponses avec un modèle spécifié
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Fournir un contexte et poser une question
context = "La Tour Eiffel est située à Paris. C'est l'un des monuments les plus célèbres au monde."
question = "Où se trouve la Tour Eiffel ?"

# Obtenir une réponse
result = qa_pipeline(question=question, context=context)
print(f"Réponse : {result['answer']}")
