import requests
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
import pocketbase

# Initialize PocketBase
pb = pocketbase.PocketBase("https://dauys.pockethost.io/")

# Register Attention Layer for Model Loading
@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="attention_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        e = tf.keras.backend.squeeze(e, axis=-1)
        alpha = tf.keras.backend.softmax(e)
        context = tf.keras.backend.sum(x * tf.keras.backend.expand_dims(alpha, axis=-1), axis=1)
        return context

# Load model with registered Attention layer
model = load_model("model1.keras", custom_objects={"Attention": Attention})

# Gesture classes (should match training labels)
gestures = ["rahmet", "sau_bol", "keshir", "salem"]

# Initialize Mediapipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Constants
FRAME_COUNT = 30  # Expected number of frames per sequence
NUM_KEYPOINTS = 1662  # Number of features per frame

# === VIDEO DOWNLOADING ===
def download_video(video_id, filename="sign_video.mp4"):
    """Download video from PocketBase and save it locally."""
    video_entry = pb.collection("videos").get_one(video_id)
    video_url = f"https://dauys.pockethost.io/api/files/videos/{video_entry.id}/{video_entry.file}"
    
    response = requests.get(video_url)
    with open(filename, "wb") as f:
        f.write(response.content)
    
    print(f"✅ Downloaded video: {filename}")
    return filename

# === FRAME PROCESSING ===
def extract_keypoints_from_frame(frame):
    """Extract keypoints from a single frame using Mediapipe. Fill missing keypoints with 0s."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    def extract_landmarks(landmarks, num_points, dim):
        if landmarks:
            return np.array([[p.x, p.y, p.z] for p in landmarks.landmark]).flatten()
        return np.zeros(num_points * dim)  # Ensure fixed size

    pose = extract_landmarks(results.pose_landmarks, 33, 4)      # (132,)
    face = extract_landmarks(results.face_landmarks, 468, 3)     # (1404,)
    left_hand = extract_landmarks(results.left_hand_landmarks, 21, 3)  # (63,)
    right_hand = extract_landmarks(results.right_hand_landmarks, 21, 3) # (63,)

    keypoints = np.concatenate([pose, face, left_hand, right_hand])  # Fixed shape
    if keypoints.shape[0] != NUM_KEYPOINTS:
        print(f"⚠️ Keypoints shape mismatch: Expected {NUM_KEYPOINTS}, Got {keypoints.shape[0]}. Padding with 0s.")
        keypoints = np.pad(keypoints, (0, NUM_KEYPOINTS - keypoints.shape[0]), mode="constant")

    return keypoints

# === VIDEO PROCESSING ===
def process_video(video_path):
    """Extract frames and make predictions."""
    cap = cv2.VideoCapture(video_path)
    frame_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints_from_frame(frame)
        frame_sequence.append(keypoints)

        if len(frame_sequence) == FRAME_COUNT:
            break  # Stop after collecting 30 frames

    cap.release()

    # Ensure we have exactly 30 frames, pad with empty frames if needed
    while len(frame_sequence) < FRAME_COUNT:
        print(f"⚠️ Missing frames: Expected {FRAME_COUNT}, Got {len(frame_sequence)}. Padding with 0s.")
        frame_sequence.append(np.zeros(NUM_KEYPOINTS))

    # Convert to NumPy array
    frame_sequence = np.array(frame_sequence)
    print(f"✅ Final Frame Sequence Shape: {frame_sequence.shape}")  # Debugging

    # Normalize (same as training)
    frame_sequence = frame_sequence / np.linalg.norm(frame_sequence, axis=-1, keepdims=True)

    # Add batch dimension
    frame_sequence = np.expand_dims(frame_sequence, axis=0)

    # Predict
    predictions = model.predict(frame_sequence)
    softmax_scores = np.round(predictions[0], 4)  # Round for better readability

    # Get predicted label
    predicted_index = np.argmax(predictions)
    predicted_label = gestures[predicted_index]

    print(f"✅ Softmax Scores: {softmax_scores}")  # Debugging
    print(f"🎯 Predicted Sign: {predicted_label}")

    return predicted_label

# === UPDATE DATABASE ===
def update_pocketbase(video_id, prediction):
    """Update PocketBase with the predicted sign language gesture."""
    pb.collection("videos").update(video_id, {
        "status": "completed",
        "predictedSign": prediction
    })
    print(f"✅ Updated PocketBase with prediction: {prediction}")

# ==== MAIN EXECUTION ====
if __name__ == "__main__":
    # Fetch the latest video with status = "pending"
    pending_videos = pb.collection("videos").get_list(1, 10, {"filter": 'status="pending"'})

    for video in pending_videos.items:
        video_id = video.id
        print(f"🔄 Processing video ID: {video_id}")

        video_path = download_video(video_id)
        prediction = process_video(video_path)
        update_pocketbase(video_id, prediction)
