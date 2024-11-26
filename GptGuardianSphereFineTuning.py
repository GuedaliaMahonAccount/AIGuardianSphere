import os
from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Set it in the .env file.")
app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle user input and return a response from the GPT-3.5-turbo model.
    """
    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Use GPT-3.5-turbo (no fine-tuning yet)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150,
            temperature=0.7,
        )

        return jsonify({"response": response["choices"][0]["message"]["content"].strip()})
    except openai.error.OpenAIError as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/fine_tune", methods=["POST"])
def fine_tune():
    """
    Endpoint to start fine-tuning a model using a local JSONL file.
    """
    try:
        # Path to the JSONL file within the project directory
        jsonl_file = request.json.get("file_path", "").strip()
        if not jsonl_file:
            return jsonify({"error": "No file path provided"}), 400
        if not os.path.exists(jsonl_file):
            return jsonify({"error": f"The file '{jsonl_file}' does not exist."}), 400

        # Step 1: Upload the JSONL file to OpenAI
        print("Uploading training file...")
        training_file = openai.File.create(
            file=open(jsonl_file, "rb"),
            purpose="fine-tune"
        )
        print(f"File uploaded with ID: {training_file['id']}")

        # Step 2: Start fine-tuning
        print("Starting fine-tuning process...")
        fine_tune_response = openai.FineTune.create(
            training_file=training_file["id"],
            model="gpt-3.5-turbo"
        )

        return jsonify({
            "message": "Fine-tuning started.",
            "fine_tune_id": fine_tune_response["id"],
            "track_command": f"openai api fine_tunes.follow -i {fine_tune_response['id']}"
        })
    except openai.error.OpenAIError as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
