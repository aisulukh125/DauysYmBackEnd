from flask import Flask, request, jsonify
import subprocess
import os
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)  # Enable CORS so React Native can communicate with Flask

# Path to process_video.py (adjust if needed)
PROCESS_SCRIPT_PATH = os.path.abspath("process_video.py")

@app.route("/process_video", methods=["POST"])
def process_video():
    """Endpoint to trigger video processing."""
    data = request.json
    video_id = data.get("videoId")
    
    if not video_id:
        return jsonify({"error": "No video ID provided"}), 400

    # Run process_video.py with video ID in a new process
    subprocess.Popen(["python", PROCESS_SCRIPT_PATH, video_id])

    return jsonify({"message": f"Processing started for video {video_id}"}), 200

if __name__ == "__main__":
    app.run(host="192.168.1.63", port=5000, debug=True)
