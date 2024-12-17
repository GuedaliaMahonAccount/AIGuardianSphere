from waitress import serve
from GptGuardianSphereFineTuning import app
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    serve(app, host="0.0.0.0", port=port)