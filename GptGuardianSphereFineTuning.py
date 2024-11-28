import os
import uuid
from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv
from flask_cors import CORS
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Set it in the .env file.")

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# MongoDB configuration
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client.get_database("chat_db")
chat_collection = db.get_collection("chats")

@app.route("/")
def home():
    return jsonify({"message": "welcome in guardian sphere model of gpt"})

@app.route("/new-chat", methods=["POST", "OPTIONS"])
def new_chat():
    if request.method == "OPTIONS":
        return {}, 200

    try:
        username = request.json.get("username")
        title = request.json.get("title")

        if not username or not title:
            return jsonify({"error": "Username and title are required"}), 400

        chat_id = str(uuid.uuid4())

        chat_data = {
            "_id": chat_id,
            "username": username,
            "title": title,
            "messages": []
        }

        chat_collection.insert_one(chat_data)

        return jsonify({
            "chat": {
                "_id": chat_id,
                "title": title,
                "messages": []
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return {}, 200

    try:
        username = request.json.get("username")
        chat_id = request.json.get("chatId")
        user_message = request.json.get("message", "").strip()

        if not username or not user_message or not chat_id:
            return jsonify({"error": "Username, chatId and message are required"}), 400

        chat = chat_collection.find_one({"_id": chat_id, "username": username})
        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        # Obtenir l'historique des messages pour ce chat
        chat_messages = chat["messages"]

        # Créer le contexte pour GPT avec l'historique
        gpt_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for msg in chat_messages:
            gpt_messages.append({"role": msg["role"], "content": msg["content"]})
        gpt_messages.append({"role": "user", "content": user_message})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=gpt_messages,
            max_tokens=150,
            temperature=0.7,
        )

        ai_message = response["choices"][0]["message"]["content"].strip()

        # Sauvegarder les nouveaux messages
        chat_messages.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_message}
        ])

        chat_collection.update_one(
            {"_id": chat_id, "username": username},
            {"$set": {"messages": chat_messages}}
        )

        return jsonify({
            "response": ai_message,
            "messages": chat_messages
        })

    except Exception as e:
        if "quota" in str(e).lower():
            return jsonify({"error": "Quota exceeded. Please check your OpenAI account for details."}), 429
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500

@app.route("/history/<username>", methods=["GET", "OPTIONS"])
def get_chat_history(username):
    if request.method == "OPTIONS":
        return {}, 200

    try:
        chats = list(chat_collection.find({"username": username}))
        history = [
            {
                "_id": chat["_id"],
                "title": chat["title"],
                "messages": chat["messages"]
            }
            for chat in chats
        ]
        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete-chat", methods=["DELETE", "OPTIONS"])
def delete_chat():
    if request.method == "OPTIONS":
        return {}, 200

    try:
        username = request.json.get("username")
        chat_id = request.json.get("chatId")

        if not username or not chat_id:
            return jsonify({"error": "Username and chatId are required"}), 400

        result = chat_collection.delete_one({"_id": chat_id, "username": username})
        if result.deleted_count == 0:
            return jsonify({"error": "Chat not found"}), 404

        return jsonify({"message": "Chat deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)