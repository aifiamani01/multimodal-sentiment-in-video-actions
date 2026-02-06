# experiments/audio_embeddings.py

import os
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
from tqdm import tqdm

# -------------------------------
# CONFIGURATION
# -------------------------------
audio_root = "data/raw/audio"           # Path to actor folders
output_root = "experiments/audio_embeddings"  # Where embeddings will be saved
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face Wav2Vec2 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.to(device)
model.eval()

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def extract_embedding(audio_path):
    # Load audio with librosa (keep original sample rate)
    waveform, sr = librosa.load(audio_path, sr=None)

    # Resample to 16kHz (Wav2Vec2 requirement)
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Convert to torch tensor using processor
    input_values = processor(waveform, sampling_rate=sr, return_tensors="pt").input_values
    input_values = input_values.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_values)
        embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (time, features)
    return embeddings


# -------------------------------
# MAIN LOOP
# -------------------------------
if not os.path.exists(output_root):
    os.makedirs(output_root)

actors = sorted(os.listdir(audio_root))  # Actor_01 ... Actor_24

for actor in actors:
    actor_path = os.path.join(audio_root, actor)
    output_actor_path = os.path.join(output_root, actor)
    os.makedirs(output_actor_path, exist_ok=True)

    audio_files = [f for f in os.listdir(actor_path) if f.endswith(".wav")]

    print(f"Processing {actor}: {len(audio_files)} files")
    for audio_file in tqdm(audio_files):
        audio_path = os.path.join(actor_path, audio_file)
        embedding = extract_embedding(audio_path)

        # Save embedding
        save_path = os.path.join(output_actor_path, audio_file.replace(".wav", ".npy"))
        np.save(save_path, embedding)

print("✅ Audio embeddings extraction complete!")
