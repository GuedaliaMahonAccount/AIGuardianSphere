import os
import uuid
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from pymongo import MongoClient
import requests
import json

# Load emergency data from JSON file
with open("emergency_numbers.json", "r", encoding="utf-8") as f:
    emergency_data = json.load(f)


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





#
#
#pre-chat functions
#
#
#detect categorie and role
def detect_role(user_message, language="en"):
    """
    Detects the role/category of the user based on the message.
    Supports multiple languages with predefined keywords.
    """
    # Define keywords for each role in different languages
    keywords = {
        "stress": [
            "stressed", "overwhelmed", "burnt out", "לחוץ", "חרדה", "anxious",
            "pressure", "under pressure", "can't relax", "מתוח", "tension"
        ],
        "depression": [
            "hopeless", "sad", "worthless", "דיכאון", "עצוב", "empty", "lost",
            "I can't go on", "helpless", "אין לי תקווה", "בדידות", "lonely"
        ],
        "anger": [
            "angry", "frustrated", "furious", "annoyed", "כועס", "עצבני",
            "rage", "irritated", "mad", "can't control myself", "זעם", "מתפרץ"
        ],
        "trauma": [
            "trauma", "triggered", "טראומה", "מופעל", "flashback", "painful memory",
            "I'm reminded of", "scared because of my past", "טריגר"
        ],
        "fear": [
            "scared", "afraid", "מפחד", "פחד", "terrified", "panicked",
            "I'm in danger", "I feel unsafe", "I'm nervous", "פאניקה", "חרדה", "panic"
        ]
    }

    # Detect role by matching keywords
    for role, words in keywords.items():
        if any(word in user_message.lower() for word in words):
            return role

    # Default role if no keywords match
    return None
def save_role_to_db(chat_id, username, role):
    """
    Saves or updates the role in the database for the given chat session.
    """
    chat_collection.update_one(
        {"_id": chat_id, "username": username},
        {"$set": {"role": role}}
    )
def generate_role_based_response(role, ai_response):
    """
    Modifies the AI response to include role-based guidance for tone and approach.
    """
    role_guidelines = {
        "stress": "Focus on calming the user and suggesting relaxation techniques.",
        "depression": "Provide empathetic support, acknowledge their feelings, and remind them they are not alone.",
        "anger": "Validate their anger and help them channel it into something constructive.",
        "trauma": "Be sensitive and encourage the user to talk in a safe space without judgment.",
        "fear": "Reassure the user and guide them through grounding or safety exercises."
    }

    # Append role-specific guidelines to the AI's main response
    if role in role_guidelines:
        return f"[Guideline: {role_guidelines[role]}] {ai_response}"
    return ai_response

#stay on topic
def stay_on_topic(user_message, current_topic):
    """
    Checks if the user message is off-topic and redirects the conversation back to the main topic.
    """
    unrelated_response = f"Let’s get back to discussing {current_topic}."
    # Example logic for detecting unrelated topics
    if is_unrelated(user_message):  # Define your logic for unrelated content
        return unrelated_response
    return None
def is_unrelated(user_message, related_keywords=None):
    """
    Checks if a message is unrelated based on keywords and context.
    """
    if related_keywords is None:
        related_keywords = [
            "stress", "anxiety", "depression", "anger", "trauma", "fear",
            "calm", "relax", "mental health", "help", "sadness", "panic"
        ]

    unrelated_keywords = ["weather", "sports", "politics", "movies", "news"]

    # Check for unrelated keywords
    if any(word in user_message.lower() for word in unrelated_keywords):
        return True

    # Check if the message contains related keywords
    if any(word in user_message.lower() for word in related_keywords):
        return False

    # Default: treat as unrelated if no match
    return True

#emergency
def check_emergency(user_message):
    user_message = user_message.lower().strip()
    cumulative_weight = 0
    threshold = 3

    emergency_keywords = emergency_data["emergency"]["keywords"]
    print(f"Checking message: {user_message}")  # Log user message


    for keyword, weight in emergency_keywords.items():
        if keyword in user_message:
            cumulative_weight += weight
            print(f"Matched keyword: {keyword}, weight: {weight}, cumulative: {cumulative_weight}")
            if cumulative_weight >= threshold:
                print("Emergency detected!")
                return True
    print("No emergency detected.")
    return False
def emergency_response(country_code="default", language="en"):
    """
    Provides an emergency response with a relevant contact number from JSON
    and supports multiple languages for the message.
    """
    numbers_by_country = emergency_data["emergency"]["numbers_by_country"]
    default_number = emergency_data["emergency"]["default_number"]

    # Fetch the country-specific number or fallback to default
    emergency_number = numbers_by_country.get(country_code.upper(), default_number)

    # Multi-language support
    messages = {
        "en": f"If you are in immediate danger, please contact emergency services or call {emergency_number}.",
        "he": f"אם אתה בסכנה מיידית, אנא צור קשר עם שירותי חירום או התקשר ל-{emergency_number}."
    }

    # Return the message in the requested language, default to English if unavailable
    return messages.get(language, messages["en"])

def detect_language(text):
    """
    Detects if the text is in Hebrew or English based on character set.
    Returns 'he' for Hebrew, 'en' for English.
    """
    if any("\u0590" <= char <= "\u05FF" for char in text):  # Hebrew Unicode range
        return "he"
    return "en"







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
        # Log start of the request
        print("Starting chat request...")

        # Get data from the request
        username = request.json.get("username")
        chat_id = request.json.get("chatId")
        user_message = request.json.get("message", "").strip()
        country_code = request.json.get("country", "default")  # Country code for emergency response

        # Log the received data
        print("Received data:", {"username": username, "chat_id": chat_id, "user_message": user_message})

        # Validate input
        if not username or not user_message or not chat_id:
            print("Missing required fields.")
            return jsonify({"error": "Username, chatId, and message are required"}), 400

        # Find chat in the database
        chat = chat_collection.find_one({"_id": chat_id, "username": username})
        if not chat:
            print("Chat not found in the database.")
            return jsonify({"error": "Chat not found"}), 404

        # Detect language of the user message
        language = detect_language(user_message)
        print(f"Detected language: {language}")

        # Detect emergency
        if check_emergency(user_message):
            print("Emergency detected. Generating emergency response.")

            # Generate the emergency response message
            emergency_message = emergency_response(country_code, language)

            # Append the emergency message to the chat history
            chat_messages = chat.get("messages", [])
            chat_messages.append({"role": "assistant", "content": emergency_message})

            # Save the updated chat messages in the database
            chat_collection.update_one(
                {"_id": chat_id, "username": username},
                {"$set": {"messages": chat_messages}}
            )

            # Return the emergency response
            return jsonify({
                "response": emergency_message,  # The emergency message
                "messages": chat_messages  # The updated chat history
            }), 200

        # Detect user role (non-emergency)
        role = detect_role(user_message)
        print(f"Detected role: {role}")
        if role:
            save_role_to_db(chat_id, username, role)
            print(f"Role {role} saved to database.")

        # Get the message history
        chat_messages = chat.get("messages", [])
        print("Chat messages:", chat_messages)

        # Prepare GPT context
        gpt_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for msg in chat_messages:
            gpt_messages.append({"role": msg["role"], "content": msg["content"]})
        gpt_messages.append({"role": "user", "content": user_message})

        # Azure OpenAI API call
        print("Preparing to send request to Azure OpenAI API...")
        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY,
        }
        payload = {
            "messages": gpt_messages,
            "max_tokens": 150,
            "temperature": 0.7,
        }

        # Log the payload being sent
        print("Payload to Azure API:", payload)

        # Send the request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        # Parse the response
        print("Azure OpenAI API responded successfully.")
        ai_message = response.json()["choices"][0]["message"]["content"].strip()
        print("AI Message:", ai_message)

        # Generate role-based response if applicable
        if role:
            ai_message = generate_role_based_response(role, ai_message)

        # Save the new messages
        chat_messages.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_message}
        ])
        chat_collection.update_one(
            {"_id": chat_id, "username": username},
            {"$set": {"messages": chat_messages}}
        )
        print("Messages saved to database.")

        # Return the AI response
        return jsonify({
            "response": ai_message,  # The AI-generated response
            "messages": chat_messages  # The updated chat history
        })

    except requests.exceptions.RequestException as azure_error:
        print("Azure API error:", str(azure_error))
        return jsonify({"error": f"Azure OpenAI API error: {str(azure_error)}"}), 500
    except Exception as general_error:
        print("Internal server error:", str(general_error))
        return jsonify({"error": f"Internal server error: {str(general_error)}"}), 500

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