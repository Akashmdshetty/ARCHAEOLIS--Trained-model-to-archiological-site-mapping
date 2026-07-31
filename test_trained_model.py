"""
test_trained_model.py
---------------------
Tests the 94.32% satellite classifier and full ARCHAEOLIS analysis pipeline
on random held-out test set images from all 5 classes (cloudy, desert, green_area, vehicles, water)
and user images.
"""

import os
import sys
import random
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.inference import ArchaeologicalAnalyzer

def test_inference_pipeline():
    print("="*65)
    print(" [TEST] ARCHAEOLIS 94.32% SATELLITE AI CLASSIFIER & INFERENCE ")
    print("="*65)

    analyzer = ArchaeologicalAnalyzer()
    test_dir = os.path.join(PROJECT_ROOT, "data", "sat_data_split", "test")
    
    classes = ["cloudy", "desert", "green_area", "vehicles", "water"]
    
    total_tested = 0
    total_correct = 0

    for cls in classes:
        cls_folder = os.path.join(test_dir, cls)
        if not os.path.exists(cls_folder):
            continue
        
        sample_files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not sample_files:
            continue
        
        # Pick 3 random test images per class
        samples = random.sample(sample_files, min(3, len(sample_files)))
        print(f"\n--- Testing Class Folder: [{cls.upper()}] ---")
        
        for fname in samples:
            img_path = os.path.join(cls_folder, fname)
            pil_img = Image.open(img_path).convert('RGB')
            tensor = analyzer.transform(pil_img).unsqueeze(0).to(analyzer.device)
            
            with torch.no_grad():
                logits = analyzer.full_sat_model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                pred_cls = classes[pred_idx]
                confidence = probs[pred_idx] * 100.0

            total_tested += 1
            is_correct = (pred_cls == cls)
            if is_correct:
                total_correct += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"

            print(f" {status} Image: {fname:<18} | True: {cls:<10} | Pred: {pred_cls:<10} ({confidence:.1f}%)")

    accuracy = (total_correct / max(total_tested, 1)) * 100.0
    print("\n" + "="*65)
    print(f" [RESULT] SAMPLE TEST ACCURACY: {accuracy:.1f}% ({total_correct}/{total_tested} Correct)")
    print("="*65)

if __name__ == "__main__":
    test_inference_pipeline()
