import cv2
import numpy as np
import torch
from PIL import Image

def overlay_mask(image, mask, color=(0, 0, 255), alpha=0.5):
    """
    Overlays a binary mask onto an image with a specific color.
    image: numpy array (H, W, 3)
    mask: numpy array (H, W) binary
    color: tuple (B, G, R)
    """
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def draw_boxes(image, boxes, confidence_threshold=0.5, color=(255, 0, 0)):
    """
    Draws blue bounding boxes for artifact detection.
    image: numpy array (H, W, 3)
    boxes: tensor [conf, x, y, w, h] or list of boxes
    """
    img_h, img_w = image.shape[:2]
    # For simulation/demonstration, we assume boxes are normalized [0, 1]
    for box in boxes:
        conf, x, y, w, h = box
        if conf > confidence_threshold:
            x1 = int((x - w/2) * img_w)
            y1 = int((y - h/2) * img_h)
            x2 = int((x + w/2) * img_w)
            y2 = int((y + h/2) * img_h)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"Artifact {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def overlay_heatmap(image, heatmap, alpha=0.6):
    """
    Overlays a yellow erosion risk heatmap.
    image: numpy array (H, W, 3)
    heatmap: numpy array (H, W) normalized [0, 1]
    """
    # Create yellow colormap (Yellow is G+R)
    # 0 -> [0,0,0], 1 -> [0, 255, 255] in BGR
    heatmap_color = np.zeros_like(image)
    heatmap_color[:, :, 1] = (heatmap * 255).astype(np.uint8) # Green
    heatmap_color[:, :, 2] = (heatmap * 255).astype(np.uint8) # Red
    
    # Mask where heatmap is significant
    mask = heatmap > 0.1
    result = image.copy()
    result[mask] = cv2.addWeighted(image[mask], 1-alpha, heatmap_color[mask], alpha, 0)
    return result

def get_placeholder_analytics(img_size=(224, 224), seed=42):
    """
    Generates deterministic dummy data based on a seed for UI demonstration.
    """
    np.random.seed(seed)
    
    # Dummy ruins (Red) - random circles
    ruins_mask = np.zeros(img_size, dtype=np.uint8)
    for _ in range(np.random.randint(2, 5)):
        center = (np.random.randint(0, img_size[1]), np.random.randint(0, img_size[0]))
        radius = np.random.randint(10, 40)
        cv2.circle(ruins_mask, center, radius, 255, -1)
    
    # Dummy vegetation (Green) - random rectangles
    veg_mask = np.zeros(img_size, dtype=np.uint8)
    for _ in range(np.random.randint(1, 3)):
        x1 = np.random.randint(0, img_size[1]//2)
        y1 = np.random.randint(0, img_size[0]//2)
        x2 = x1 + np.random.randint(50, 150)
        y2 = y1 + np.random.randint(50, 150)
        cv2.rectangle(veg_mask, (x1, y1), (x2, y2), 255, -1)
    
    # Dummy detection (Blue boxes)
    boxes = []
    for _ in range(np.random.randint(1, 4)):
        boxes.append([
            np.random.uniform(0.7, 0.99), # confidence
            np.random.uniform(0.1, 0.9),  # x
            np.random.uniform(0.1, 0.9),  # y
            np.random.uniform(0.05, 0.2), # w
            np.random.uniform(0.05, 0.2)  # h
        ])
    
    # Dummy erosion (Yellow heatmap) - random gradient
    erosion_heatmap = np.zeros(img_size, dtype=np.float32)
    start_point = np.random.uniform(0, 1), np.random.uniform(0, 1)
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            norm_i, norm_j = i/img_size[0], j/img_size[1]
            dist = np.sqrt((norm_i - start_point[0])**2 + (norm_j - start_point[1])**2)
            erosion_heatmap[i, j] = np.clip(1.0 - dist, 0, 1)
            
    # Dummy faults (Purple) - random lines
    fault_mask = np.zeros(img_size, dtype=np.uint8)
    for _ in range(np.random.randint(1, 3)):
        pt1 = (np.random.randint(0, img_size[1]), np.random.randint(0, img_size[0]))
        pt2 = (np.random.randint(0, img_size[1]), np.random.randint(0, img_size[0]))
        cv2.line(fault_mask, pt1, pt2, 255, np.random.randint(1, 3))

    return ruins_mask, veg_mask, boxes, erosion_heatmap, fault_mask

def process_multi_task_results(results, img_size=(224, 224)):
    """
    Converts raw model tensors from MultiTaskArchaeologist into numpy masks/boxes.
    """
    # 1. Segmentation (B, 3, H, W) -> ruins and veg masks
    seg_logits = results['segmentation']
    seg_probs = torch.softmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
    
    # Class 1: Ruins, Class 2: Vegetation (assuming index 0 is background)
    ruins_mask = (seg_probs[1] > 0.5).astype(np.uint8) * 255
    veg_mask = (seg_probs[2] > 0.5).astype(np.uint8) * 255
    
    # 2. Erosion Heatmap (B, 1, H, W) -> (H, W)
    erosion_heatmap = results['erosion'].squeeze().cpu().numpy()
    
    # 3. Fault Mask (B, 1, H, W) -> (H, W)
    fault_mask = (results['faults'].squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    
    # 4. Detection (B, 5, H_feat, W_feat)
    # Simplified: find peaks in detection confidence
    det_tensor = results['detection'].squeeze(0).cpu().numpy() # (5, 7, 7) or similar
    conf_map = det_tensor[0]
    boxes = []
    
    # Simple threshold-based box extraction from grid
    h_feat, w_feat = conf_map.shape
    for i in range(h_feat):
        for j in range(w_feat):
            conf = conf_map[i, j]
            if conf > 0.5:
                # Local grid-based coordinates
                x, y, w, h = det_tensor[1:, i, j]
                # Map relative to grid cell
                abs_x = (j + x) / w_feat
                abs_y = (i + y) / h_feat
                boxes.append([conf, abs_x, abs_y, w, h])
    
    return ruins_mask, veg_mask, boxes, erosion_heatmap, fault_mask
