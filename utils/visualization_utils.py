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

def create_satellite_scanner_composite(base_rgb, ruins_mask, veg_mask, erosion_map, faults_mask, artifacts, place_name, lat, lon, radius_km,
                                       show_ruins=True, show_veg=True, show_erosion=True, show_faults=True, show_artifacts=True, show_hud=True):
    """
    Creates a multi-layered high-tech satellite scan composite image.
    Layer 1: Real satellite base image (RGB) - preserved with high clarity
    Layer 2: Multi-spectral Vegetation (NDVI Emerald Green translucent)
    Layer 3: Soil Erosion LiDAR / Thermal Heatmap (Yellow/Orange subtle hotspots)
    Layer 4: Structural Ruin Contours & Foundations (Neon Red outlines)
    Layer 5: Geological Fault Lineaments (Violet/Purple fracture lines)
    Layer 6: Artifact Bounding Boxes (Neon Blue)
    Layer 7: Cybernetic Scanner HUD (GPS Coords, Target Reticle, Scale Brackets)
    """
    h, w = base_rgb.shape[:2]
    composite = base_rgb.copy()
    
    # 1. Vegetation Layer (Emerald Green - Translucent Tint)
    if show_veg and veg_mask is not None:
        veg_bin = (veg_mask > 128).astype(np.uint8) if veg_mask.dtype != np.uint8 else (veg_mask > 0).astype(np.uint8)
        if veg_bin.any():
            veg_layer = composite.copy()
            veg_layer[veg_bin > 0] = [0, 230, 118]
            composite[veg_bin > 0] = cv2.addWeighted(veg_layer[veg_bin > 0], 0.35, composite[veg_bin > 0], 0.65, 0)

    # 2. Erosion Risk Thermal Heatmap Layer (Yellow/Orange Thermal Hotspot Marks)
    if show_erosion and erosion_map is not None:
        er_norm = cv2.resize(erosion_map.astype(np.float32), (w, h))
        er_min, er_max = er_norm.min(), er_norm.max()
        if er_max - er_min > 0.001:
            er_scaled = (er_norm - er_min) / (er_max - er_min)
        else:
            er_scaled = np.zeros_like(er_norm)
            
        mask_er = er_scaled > 0.68
        if mask_er.any():
            er_color = np.zeros((h, w, 3), dtype=np.uint8)
            er_color[:, :, 0] = np.uint8(np.clip(er_scaled * 255, 0, 255))
            er_color[:, :, 1] = np.uint8(np.clip(er_scaled * 170, 0, 255))
            er_color[:, :, 2] = 0
            composite[mask_er] = cv2.addWeighted(er_color[mask_er], 0.40, composite[mask_er], 0.60, 0)
            
            er_bin = (er_scaled > 0.72).astype(np.uint8)
            contours_er, _ = cv2.findContours(er_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(composite, contours_er, -1, (255, 170, 0), 2)

    # 3. Structural Ruin Contour & Foundation Layer (Neon Red Building Foundations & Contours)
    if show_ruins and ruins_mask is not None:
        ruin_bin = (ruins_mask > 128).astype(np.uint8) if ruins_mask.dtype != np.uint8 else (ruins_mask > 0).astype(np.uint8)
        if ruin_bin.any():
            ruin_layer = composite.copy()
            ruin_layer[ruin_bin > 0] = [255, 45, 85]
            composite[ruin_bin > 0] = cv2.addWeighted(ruin_layer[ruin_bin > 0], 0.45, composite[ruin_bin > 0], 0.55, 0)
            
            contours, _ = cv2.findContours(ruin_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(composite, contours, -1, (255, 255, 255), 2)
            cv2.drawContours(composite, contours, -1, (255, 45, 85), 2)

    # 4. Geological Fault Fractures (Purple/Violet Lineament Marks & Fracture Outlines)
    if show_faults and faults_mask is not None:
        f_norm = cv2.resize(faults_mask.astype(np.float32), (w, h))
        f_min, f_max = f_norm.min(), f_norm.max()
        if f_max - f_min > 0.001:
            f_scaled = (f_norm - f_min) / (f_max - f_min)
        else:
            f_scaled = np.zeros_like(f_norm)
            
        mask_f = f_scaled > 0.70
        if mask_f.any():
            fault_layer = composite.copy()
            fault_layer[mask_f] = [175, 82, 222]
            composite[mask_f] = cv2.addWeighted(fault_layer[mask_f], 0.45, composite[mask_f], 0.55, 0)
            
            f_bin = (f_scaled > 0.75).astype(np.uint8)
            contours_f, _ = cv2.findContours(f_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(composite, contours_f, -1, (220, 120, 255), 2)

    # 5. Artifact Signal Detection Boxes
    if show_artifacts and artifacts:
        for box in artifacts:
            conf, x, y, bw, bh = box
            if conf > 0.35:
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)
                cv2.rectangle(composite, (x1, y1), (x2, y2), (0, 229, 255), 2)
                cv2.putText(composite, f"ART-SIG {conf*100:.0f}%", (x1, max(y1-5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 229, 255), 1)

    # 6. High-Tech Satellite Scanner HUD Overlay
    if show_hud:
        cx, cy = w // 2, h // 2
        cv2.drawMarker(composite, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 40, 1)
        cv2.circle(composite, (cx, cy), 60, (0, 255, 255), 1)
        cv2.circle(composite, (cx, cy), 180, (0, 229, 255), 1)

        corner_len = 35
        # Top-Left
        cv2.line(composite, (20, 20), (20 + corner_len, 20), (0, 229, 255), 2)
        cv2.line(composite, (20, 20), (20, 20 + corner_len), (0, 229, 255), 2)
        # Top-Right
        cv2.line(composite, (w - 20, 20), (w - 20 - corner_len, 20), (0, 229, 255), 2)
        cv2.line(composite, (w - 20, 20), (w - 20, 20 + corner_len), (0, 229, 255), 2)
        # Bottom-Left
        cv2.line(composite, (20, h - 20), (20 + corner_len, h - 20), (0, 229, 255), 2)
        cv2.line(composite, (20, h - 20), (20, h - 20 - corner_len), (0, 229, 255), 2)
        # Bottom-Right
        cv2.line(composite, (w - 20, h - 20), (w - 20 - corner_len, h - 20), (0, 229, 255), 2)
        cv2.line(composite, (w - 20, h - 20), (w - 20, h - 20 - corner_len), (0, 229, 255), 2)

        hud_bar = np.zeros((45, w, 3), dtype=np.uint8)
        composite[:45] = cv2.addWeighted(composite[:45], 0.35, hud_bar, 0.65, 0)
        
        info_text = f"SATELLITE SCANNER // {place_name} ({lat:.4f}N, {lon:.4f}E) | ZONE: {radius_km}KM"
        cv2.putText(composite, info_text, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return composite

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
