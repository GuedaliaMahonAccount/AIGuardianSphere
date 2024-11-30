import os
import uuid
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from pymongo import MongoClient
import requests

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION]):
    raise ValueError("Azure OpenAI configuration is incomplete. Check your .env file.")

# Flask application
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
    return jsonify({"message": "Welcome to the Azure OpenAI GPT-powered chat application"})

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

        # Get the message history for this chat
        chat_messages = chat["messages"]

        # Create the context for GPT with the history
        gpt_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for msg in chat_messages:
            gpt_messages.append({"role": msg["role"], "content": msg["content"]})
        gpt_messages.append({"role": "user", "content": user_message})

        # Azure OpenAI API call
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/completions?api-version={AZURE_OPENAI_API_VERSION}"
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY,
        }
        payload = {
            "messages": gpt_messages,
            "max_tokens": 150,
            "temperature": 0.7,
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        ai_message = response.json()["choices"][0]["message"]["content"].strip()

        # Save the new messages
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

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Azure OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

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

@app.route("/update-chat-title", methods=["PUT", "OPTIONS"])
def update_chat_title():
    if request.method == "OPTIONS":
        return {}, 200

    try:
        username = request.json.get("username")
        chat_id = request.json.get("chatId")
        new_title = request.json.get("newTitle")

        if not username or not chat_id or not new_title:
            return jsonify({"error": "Username, chatId, and newTitle are required"}), 400

        result = chat_collection.update_one(
            {"_id": chat_id, "username": username},
            {"$set": {"title": new_title}}
        )

        if result.matched_count == 0:
            return jsonify({"error": "Chat not found"}), 404

        return jsonify({"message": "Chat title updated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
