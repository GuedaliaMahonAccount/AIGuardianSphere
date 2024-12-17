from waitress import serve
from GptGuardianSphereFineTuning import app
import sys
import traceback

if __name__ == "__main__":
    try:
        print(f"Python Path: {sys.path}")
        print(f"Python Version: {sys.version}")
        serve(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Startup Error: {e}")
        traceback.print_exc()