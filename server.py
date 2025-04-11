from flask import Flask, request, jsonify
from supabase import create_client, Client
from flask_cors import CORS
import os
from dotenv import load_dotenv
import threading

app = Flask(__name__)
CORS(app)  # Enable CORS so React Native can communicate with Flask

load_dotenv()
# Load environment variables from .env file

SUPABASE_URL = os.getenv("REACT_APP_SUPABASE_URL")
SUPABASE_KEY = os.getenv("REACT_APP_SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Path to process_video.py (adjust if needed)
PROCESS_SCRIPT_PATH = os.path.abspath("process_video.py")

@app.route("/process_video", methods=["POST"])
def process_video():
    """Endpoint to trigger video processing."""
    data = request.json
    recognition_id = data.get("recognitionId")
    
    if not recognition_id:
        return jsonify({"error": "No recognition ID provided"}), 400

    # Run process_video.py in a background thread
    threading.Thread(target=process_video_in_background, args=(recognition_id,)).start()

    return jsonify({"message": f"Processing started for recognition ID {recognition_id}"}), 200

def process_video_in_background(recognition_id):
    """Process video in the background."""
    # Logic to fetch, process, and update the video from Supabase
    from process_video import download_video, process_video, update_recognition_status

    try:
        # Download video
        video_path = download_video(recognition_id)
        
        # Process video and make predictions
        prediction = process_video(video_path)
        
        # Update Supabase with prediction result
        update_recognition_status(recognition_id, prediction)
    except Exception as e:
        print(f"❌ Error processing video: {e}")

if __name__ == "__main__":
    app.run(host="192.168.1.63", port=5000, debug=True)
