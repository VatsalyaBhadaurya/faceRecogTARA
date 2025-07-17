import cv2
import torch
from torchvision import transforms
from face_alignment import align
from backbones import get_model
import numpy as np
import os
from collections import defaultdict

# --- CONFIG ---
MODEL_NAME = "edgeface_xxs"  # Use the smallest model for minimal GPU usage
CHECKPOINT_PATH = f"checkpoints/{MODEL_NAME}.pt"
FACE_DB_DIR = "face_db"  # Directory with subfolders for each person, each containing face images
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- UTILS ---
def load_face_db(face_db_dir, model, transform):
    """Load face database: returns dict {name: [embedding, ...]}"""
    db = defaultdict(list)
    for person in os.listdir(face_db_dir):
        person_dir = os.path.join(face_db_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                aligned = align.get_aligned_face(img_path)
                if aligned is None:
                    continue
                emb = model(transform(aligned).unsqueeze(0).to(DEVICE)).detach().cpu().numpy().flatten()
                db[person].append(emb)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
    # Average embeddings per person
    db_avg = {k: np.mean(v, axis=0) for k, v in db.items() if len(v) > 0}
    return db_avg

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(embedding, db, threshold=0.5):
    best_name = "Unknown"
    best_score = threshold
    for name, db_emb in db.items():
        score = cosine_similarity(embedding, db_emb)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score

# --- MAIN ---
def main():
    # Load model
    model = get_model(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load face database
    if not os.path.exists(FACE_DB_DIR):
        print(f"Face DB directory '{FACE_DB_DIR}' not found. Please create it with subfolders for each person.")
        return
    print("Loading face database...")
    face_db = load_face_db(FACE_DB_DIR, model, transform)
    print(f"Loaded {len(face_db)} identities.")

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    print("Press 'q' to quit.")
    frame_count = 0
    last_annotated_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 != 0:
            if last_annotated_frame is not None:
                cv2.imshow('EdgeFace eduTARA Recognition', last_annotated_frame)
            else:
                cv2.imshow('EdgeFace eduTARA Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = align.Image.fromarray(rgb)
        faces = []
        try:
            _, faces = align.mtcnn_model.align_multi(pil_img, limit=5)
        except Exception as e:
            pass
        for i, face in enumerate(faces):
            x, y, w, h = 10, 10 + i*120, 112, 112  # Just for display, not real bbox
            emb = model(transform(face).unsqueeze(0).to(DEVICE)).detach().cpu().numpy().flatten()
            name, score = recognize_face(emb, face_db)
            # Draw face and label
            face_np = np.array(face)
            frame[y:y+h, x:x+w] = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"{name} ({score:.2f})", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        last_annotated_frame = frame.copy()
        cv2.imshow('EdgeFace eduTARA Recognition', last_annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 