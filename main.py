import os
import cv2
import numpy as np
from deepface import DeepFace
import torch
from torch.nn import functional as F
import json
from pathlib import Path
from torchvision import transforms

# Import FER module - install with 'uv pip install fer' if not available
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("Warning: FER module not found. Install with 'uv pip install fer' for enhanced face detection.")
    print("Continuing with DeepFace only...")

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

EMO_LABELS = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
EMO_MIN_TOP = float(os.environ.get('EMO_MIN_TOP', 0.35))
EMO_T = float(os.environ.get('EMO_T', 1.0))
DEBUG_EMO = os.environ.get('DEBUG_EMO', '0') == '1'

# Custom model paths
MODEL_DIR = Path(__file__).resolve().parent / 'models'
MODEL_PATH = MODEL_DIR / 'emotion_resnet18.pt'
LABELS_PATH = MODEL_DIR / 'emotion_labels.json'

# Load custom model if available
CUSTOM_MODEL_AVAILABLE = False
CUSTOM_LABELS = EMO_LABELS  # Default to standard labels

try:
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        # Load the model
        CUSTOM_MODEL = torch.jit.load(str(MODEL_PATH))
        CUSTOM_MODEL.eval()
        
        # Load the labels
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            CUSTOM_LABELS = json.load(f)
            
        CUSTOM_MODEL_AVAILABLE = True
        if DEBUG_EMO:
            print(f"DEBUG_EMO: Custom emotion model loaded successfully with labels: {CUSTOM_LABELS}")
except Exception as e:
    if DEBUG_EMO:
        print(f"DEBUG_EMO: Failed to load custom model: {str(e)}")
    CUSTOM_MODEL_AVAILABLE = False

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------

def softmax_labels(labels, d, temperature: float = 1.0):
    """Convert dict of raw scores into a softmaxed probability dict."""
    arr = np.array([float(d.get(k, 0.0)) for k in labels], dtype=np.float32)
    if temperature <= 0:
        temperature = 1.0
    scaled = arr / float(temperature)
    m = np.max(scaled) if scaled.size else 0.0
    ex = np.exp(scaled - m)
    s = np.sum(ex) + 1e-8
    p = ex / s
    return {k: float(v) for k, v in zip(labels, p)}

def renormalize_with_neutral_floor(probs: dict, neutral_label: str = 'neutral', neutral_floor: float = 0.35):
    """Ensure probability distribution is valid and assign minimum neutral floor if low confidence."""
    p = {k: float(v) for k, v in probs.items()}
    for k in EMO_LABELS:
        p.setdefault(k, 0.0)

    total = sum(p.values()) + 1e-12
    for k in p:
        p[k] /= total

    top_label, top_p = max(p.items(), key=lambda kv: kv[1])
    if top_p >= neutral_floor:
        return p, top_label, top_p

    neutral_current = p.get(neutral_label, 0.0)
    neutral_target = max(neutral_current, neutral_floor)
    other_sum = sum(v for k, v in p.items() if k != neutral_label)
    if other_sum <= 1e-9:
        p = {k: 0.0 for k in p}
        p[neutral_label] = 1.0
        return p, neutral_label, 1.0

    scale = max(0.0, (1.0 - neutral_target) / other_sum)
    for k in p:
        if k == neutral_label:
            p[k] = neutral_target
        else:
            p[k] *= scale

    total = sum(p.values()) + 1e-12
    for k in p:
        p[k] /= total

    top_label, top_p = max(p.items(), key=lambda kv: kv[1])
    return p, top_label, top_p

def predict_with_custom_model(face_img):
    """Use the custom trained model to predict emotions from a face image."""
    if not CUSTOM_MODEL_AVAILABLE:
        if DEBUG_EMO:
            print("DEBUG_EMO: Custom model not available, skipping custom prediction")
        return None
    
    try:
        # Preprocess the image to match the training preprocessing
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),  # Match the size used in training (112x112)
            transforms.ToTensor(),
            normalize,
        ])
        
        # Convert BGR to RGB (OpenCV uses BGR, PyTorch expects RGB)
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        tensor = transform(rgb_img).unsqueeze(0)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            outputs = CUSTOM_MODEL(tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
        
        # Convert to dictionary
        emotion_probs = {CUSTOM_LABELS[i]: float(probs[i]) for i in range(len(CUSTOM_LABELS))}
        
        # Apply post-processing to improve accuracy
        # Boost happy and reduce angry if they're close
        if 'happy' in emotion_probs and 'angry' in emotion_probs:
            happy_score = emotion_probs['happy']
            angry_score = emotion_probs['angry']
            
            # If happy is reasonably high but angry is higher, adjust the scores
            if happy_score > 0.2 and angry_score > happy_score and angry_score - happy_score < 0.3:
                # Boost happy by 20% and reduce angry by 20%
                emotion_probs['happy'] = min(1.0, happy_score * 1.2)
                emotion_probs['angry'] = max(0.0, angry_score * 0.8)
                
                if DEBUG_EMO:
                    print(f"DEBUG_EMO: Adjusted happy/angry scores - happy: {happy_score:.3f}->{emotion_probs['happy']:.3f}, angry: {angry_score:.3f}->{emotion_probs['angry']:.3f}")
        
        if DEBUG_EMO:
            print(f"DEBUG_EMO: Custom model prediction: {emotion_probs}")
        
        return emotion_probs
    
    except Exception as e:
        if DEBUG_EMO:
            print(f"DEBUG_EMO: Error in custom model prediction: {str(e)}")
        return None

def extract_faces(image, min_size=64):
    """Extract faces from an image using multiple detection methods.
    Enhanced to better handle multiple faces and reduce false negatives."""
    faces = []
    face_regions = []
    
    # Try OpenCV's face detector first (fast)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try with different scale factors for better detection
    scale_factors = [1.05, 1.1, 1.15]
    min_neighbors_values = [3, 4, 5]
    
    for scale in scale_factors:
        for min_neighbors in min_neighbors_values:
            opencv_faces = face_cascade.detectMultiScale(gray, scale, min_neighbors)
            
            if len(opencv_faces) > 0:
                if DEBUG_EMO:
                    print(f"DEBUG_EMO: OpenCV detected {len(opencv_faces)} faces with scale={scale}, min_neighbors={min_neighbors}")
                
                for (x, y, w, h) in opencv_faces:
                    if w >= min_size and h >= min_size:
                        # Add some margin around the face for better emotion detection
                        margin_x = int(w * 0.1)
                        margin_y = int(h * 0.1)
                        
                        # Ensure we don't go out of bounds
                        x_with_margin = max(0, x - margin_x)
                        y_with_margin = max(0, y - margin_y)
                        w_with_margin = min(image.shape[1] - x_with_margin, w + 2 * margin_x)
                        h_with_margin = min(image.shape[0] - y_with_margin, h + 2 * margin_y)
                        
                        face_img = image[y_with_margin:y_with_margin+h_with_margin, 
                                         x_with_margin:x_with_margin+w_with_margin]
                        
                        # Check if this face overlaps significantly with any existing face
                        new_face = True
                        for ex_x, ex_y, ex_w, ex_h in face_regions:
                            # Calculate overlap
                            overlap_x = max(0, min(ex_x + ex_w, x_with_margin + w_with_margin) - max(ex_x, x_with_margin))
                            overlap_y = max(0, min(ex_y + ex_h, y_with_margin + h_with_margin) - max(ex_y, y_with_margin))
                            overlap_area = overlap_x * overlap_y
                            min_area = min(ex_w * ex_h, w_with_margin * h_with_margin)
                            if overlap_area > 0.5 * min_area:  # If overlap is more than 50%
                                new_face = False
                                break
                        
                        if new_face:
                            faces.append(face_img)
                            face_regions.append((x_with_margin, y_with_margin, w_with_margin, h_with_margin))
                
                # If we found faces with this configuration, no need to try others
                if len(faces) > 0:
                    break
            
        # If we found faces with this scale factor, no need to try others
        if len(faces) > 0:
            break
    
    # If FER is available, use MTCNN for better detection
    if FER_AVAILABLE:
        try:
            fer_detector = FER(mtcnn=True)
            fer_result = fer_detector.detect_emotions(image)
            
            if DEBUG_EMO:
                print(f"DEBUG_EMO: FER MTCNN detected {len(fer_result)} faces")
            
            for face_data in fer_result:
                x, y, w, h = [int(val) for val in face_data['box']]
                if w >= min_size and h >= min_size:
                    # Add some margin around the face for better emotion detection
                    margin_x = int(w * 0.1)
                    margin_y = int(h * 0.1)
                    
                    # Ensure we don't go out of bounds
                    x_with_margin = max(0, x - margin_x)
                    y_with_margin = max(0, y - margin_y)
                    w_with_margin = min(image.shape[1] - x_with_margin, w + 2 * margin_x)
                    h_with_margin = min(image.shape[0] - y_with_margin, h + 2 * margin_y)
                    
                    face_img = image[y_with_margin:y_with_margin+h_with_margin, 
                                     x_with_margin:x_with_margin+w_with_margin]
                    
                    # Check if this face overlaps significantly with any existing face
                    new_face = True
                    for ex_x, ex_y, ex_w, ex_h in face_regions:
                        # Calculate overlap
                        overlap_x = max(0, min(ex_x + ex_w, x_with_margin + w_with_margin) - max(ex_x, x_with_margin))
                        overlap_y = max(0, min(ex_y + ex_h, y_with_margin + h_with_margin) - max(ex_y, y_with_margin))
                        overlap_area = overlap_x * overlap_y
                        min_area = min(ex_w * ex_h, w_with_margin * h_with_margin)
                        if overlap_area > 0.5 * min_area:  # If overlap is more than 50%
                            new_face = False
                            break
                    
                    if new_face:
                        faces.append(face_img)
                        face_regions.append((x_with_margin, y_with_margin, w_with_margin, h_with_margin))
        except Exception as e:
            if DEBUG_EMO:
                print(f"DEBUG_EMO: FER face detection error: {str(e)}")
    
    # Try one more method if we still have few or no faces
    if len(faces) < 2:
        try:
            # Try with a different OpenCV cascade for profile faces
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            profile_faces = profile_cascade.detectMultiScale(gray, 1.1, 4)
            
            if DEBUG_EMO:
                print(f"DEBUG_EMO: OpenCV profile detector found {len(profile_faces)} faces")
            
            for (x, y, w, h) in profile_faces:
                if w >= min_size and h >= min_size:
                    # Add some margin
                    margin_x = int(w * 0.1)
                    margin_y = int(h * 0.1)
                    
                    # Ensure we don't go out of bounds
                    x_with_margin = max(0, x - margin_x)
                    y_with_margin = max(0, y - margin_y)
                    w_with_margin = min(image.shape[1] - x_with_margin, w + 2 * margin_x)
                    h_with_margin = min(image.shape[0] - y_with_margin, h + 2 * margin_y)
                    
                    face_img = image[y_with_margin:y_with_margin+h_with_margin, 
                                     x_with_margin:x_with_margin+w_with_margin]
                    
                    # Check for overlap
                    new_face = True
                    for ex_x, ex_y, ex_w, ex_h in face_regions:
                        overlap_x = max(0, min(ex_x + ex_w, x_with_margin + w_with_margin) - max(ex_x, x_with_margin))
                        overlap_y = max(0, min(ex_y + ex_h, y_with_margin + h_with_margin) - max(ex_y, y_with_margin))
                        overlap_area = overlap_x * overlap_y
                        min_area = min(ex_w * ex_h, w_with_margin * h_with_margin)
                        if overlap_area > 0.5 * min_area:
                            new_face = False
                            break
                    
                    if new_face:
                        faces.append(face_img)
                        face_regions.append((x_with_margin, y_with_margin, w_with_margin, h_with_margin))
        except Exception as e:
            if DEBUG_EMO:
                print(f"DEBUG_EMO: Profile face detection error: {str(e)}")
    
    if DEBUG_EMO:
        print(f"DEBUG_EMO: Extracted {len(faces)} faces from image")
        
        # Save individual faces for debugging if needed
        if len(faces) > 0:
            os.makedirs("debug_faces", exist_ok=True)
            for i, face in enumerate(faces):
                cv2.imwrite(f"debug_faces/face_{i+1}.jpg", face)
                print(f"DEBUG_EMO: Saved face_{i+1}.jpg for debugging")
    
    return faces, face_regions

# -------------------------------------------------------------------
# EMOTION ANALYSIS
# -------------------------------------------------------------------

def analyze_emotion_ensemble(image):
    """Perform emotion analysis combining DeepFace, FER, and custom model.
    Works with multiple faces, analyzing each face individually."""
    # Extract faces from the image
    faces, face_regions = extract_faces(image)
    face_count = len(faces)
    
    if face_count == 0:
        if DEBUG_EMO:
            print("DEBUG_EMO: No faces detected in the image")
        # Return neutral with low confidence
        neutral_probs = {k: 0.0 for k in EMO_LABELS}
        neutral_probs['neutral'] = 1.0
        return 'neutral', 0.35, neutral_probs, 0, []
    
    # Process each face
    all_face_results = []
    
    for i, (face_img, (x, y, w, h)) in enumerate(zip(faces, face_regions)):
        if DEBUG_EMO:
            print(f"DEBUG_EMO: Analyzing face {i+1}/{face_count}")
        
        face_emotions = []
        
        # 1. Try custom model first (if available)
        if CUSTOM_MODEL_AVAILABLE:
            custom_probs = predict_with_custom_model(face_img)
            if custom_probs:
                face_emotions.append({
                    'source': 'custom',
                    'emotions': custom_probs
                })
                if DEBUG_EMO:
                    print(f"DEBUG_EMO: Custom model results for face {i+1}: {custom_probs}")
        
        # 2. Try DeepFace
        try:
            df_result = DeepFace.analyze(
                face_img, 
                actions=['emotion'], 
                enforce_detection=False
            )
            
            if 'emotion' in df_result:
                face_emotions.append({
                    'source': 'deepface',
                    'emotions': df_result['emotion']
                })
                if DEBUG_EMO:
                    print(f"DEBUG_EMO: DeepFace results for face {i+1}: {df_result['emotion']}")
        except Exception as e:
            if DEBUG_EMO:
                print(f"DEBUG_EMO: DeepFace error for face {i+1}: {str(e)}")
        
        # 3. Try FER if available
        if FER_AVAILABLE:
            try:
                fer_detector = FER(mtcnn=True)
                fer_result = fer_detector.detect_emotions(face_img)
                
                if fer_result and len(fer_result) > 0:
                    face_emotions.append({
                        'source': 'fer',
                        'emotions': fer_result[0]['emotions']
                    })
                    if DEBUG_EMO:
                        print(f"DEBUG_EMO: FER results for face {i+1}: {fer_result[0]['emotions']}")
            except Exception as e:
                if DEBUG_EMO:
                    print(f"DEBUG_EMO: FER error for face {i+1}: {str(e)}")
        
        # Combine results with priority: custom > fer > deepface
        if not face_emotions:
            # No results from any method
            probs = {k: 0.0 for k in EMO_LABELS}
            probs['neutral'] = 1.0
        else:
            # Prioritize custom model if available
            custom_results = [e for e in face_emotions if e['source'] == 'custom']
            if custom_results:
                # Use custom model results with higher weight (0.7) - increased from 0.6
                custom_probs = softmax_labels(EMO_LABELS, custom_results[0]['emotions'], temperature=EMO_T)
                
                # Special handling for happy/angry confusion
                # If happy is reasonably high in custom model, boost it further
                if custom_probs.get('happy', 0) > 0.3 and custom_probs.get('angry', 0) > 0.2:
                    if DEBUG_EMO:
                        print(f"DEBUG_EMO: Detected potential happy/angry confusion in face {i+1}")
                        print(f"DEBUG_EMO: Before adjustment - happy: {custom_probs.get('happy', 0):.3f}, angry: {custom_probs.get('angry', 0):.3f}")
                    
                    # Boost happy and reduce angry
                    happy_score = custom_probs.get('happy', 0)
                    angry_score = custom_probs.get('angry', 0)
                    
                    # Apply stronger boost if happy is already significant
                    custom_probs['happy'] = min(1.0, happy_score * 1.3)
                    custom_probs['angry'] = max(0.0, angry_score * 0.7)
                    
                    # Renormalize
                    total = sum(custom_probs.values()) + 1e-8
                    for k in custom_probs:
                        custom_probs[k] /= total
                    
                    if DEBUG_EMO:
                        print(f"DEBUG_EMO: After adjustment - happy: {custom_probs.get('happy', 0):.3f}, angry: {custom_probs.get('angry', 0):.3f}")
                
                # Get other results
                other_results = [e for e in face_emotions if e['source'] != 'custom']
                if other_results:
                    # Process FER results separately as they might be more accurate for certain emotions
                    fer_results = [e for e in other_results if e['source'] == 'fer']
                    deepface_results = [e for e in other_results if e['source'] == 'deepface']
                    
                    # Process FER results
                    fer_probs = {}
                    if fer_results:
                        for result in fer_results:
                            result_probs = softmax_labels(EMO_LABELS, result['emotions'], temperature=EMO_T)
                            for k in EMO_LABELS:
                                fer_probs[k] = fer_probs.get(k, 0.0) + result_probs.get(k, 0.0)
                        
                        # Normalize FER probs
                        fer_sum = sum(fer_probs.values()) + 1e-8
                        for k in fer_probs:
                            fer_probs[k] /= fer_sum
                    
                    # Process DeepFace results
                    deepface_probs = {}
                    if deepface_results:
                        for result in deepface_results:
                            result_probs = softmax_labels(EMO_LABELS, result['emotions'], temperature=EMO_T)
                            for k in EMO_LABELS:
                                deepface_probs[k] = deepface_probs.get(k, 0.0) + result_probs.get(k, 0.0)
                        
                        # Normalize DeepFace probs
                        deepface_sum = sum(deepface_probs.values()) + 1e-8
                        for k in deepface_probs:
                            deepface_probs[k] /= deepface_sum
                    
                    # Combine all results with weighted average
                    # Custom: 70%, FER: 20%, DeepFace: 10%
                    probs = {k: 0.7 * custom_probs.get(k, 0.0) for k in EMO_LABELS}
                    
                    if fer_probs:
                        for k in EMO_LABELS:
                            probs[k] += 0.2 * fer_probs.get(k, 0.0)
                    
                    if deepface_probs:
                        for k in EMO_LABELS:
                            probs[k] += 0.1 * deepface_probs.get(k, 0.0)
                else:
                    # Only custom results available
                    probs = custom_probs
            else:
                # No custom results, average other results
                all_probs = []
                for result in face_emotions:
                    all_probs.append(softmax_labels(EMO_LABELS, result['emotions'], temperature=EMO_T))
                
                # Average all probabilities
                probs = {k: 0.0 for k in EMO_LABELS}
                for p in all_probs:
                    for k in EMO_LABELS:
                        probs[k] += p.get(k, 0.0)
                
                # Normalize
                total = sum(probs.values()) + 1e-8
                for k in probs:
                    probs[k] /= total
        
        # Normalize probabilities
        probs, top_label, top_p = renormalize_with_neutral_floor(probs, neutral_floor=EMO_MIN_TOP)
        
        # Store results for this face
        all_face_results.append({
            'region': (x, y, w, h),
            'emotion': top_label,
            'confidence': top_p,
            'probabilities': probs
        })
    
    # For backward compatibility, return results for the primary (first) face
    primary_result = all_face_results[0]
    primary_emotion = primary_result['emotion']
    primary_confidence = primary_result['confidence']
    primary_probs = primary_result['probabilities']
    
    if DEBUG_EMO:
        print(f"DEBUG_EMO: Detected {face_count} face(s) in the image")
        print(f"DEBUG_EMO: Primary face emotion: {primary_emotion} (confidence: {primary_confidence:.2f})")
        if face_count > 1:
            for i, result in enumerate(all_face_results):
                print(f"DEBUG_EMO: Face {i+1}: {result['emotion']} (confidence: {result['confidence']:.2f})")
    
    # Return results with face count and all face results
    return primary_emotion, float(primary_confidence), primary_probs, face_count, all_face_results

# -------------------------------------------------------------------
# FACE DETECTION UTILITIES
# -------------------------------------------------------------------

def draw_face_boxes(image, face_results):
    """Draw boxes around detected faces with emotion labels and enhanced visualization."""
    # Create a copy of the image for drawing
    annotated = image.copy()
    
    # Define colors for different emotions (BGR format)
    emotion_colors = {
        'happy': (0, 255, 0),      # Green
        'sad': (255, 0, 0),        # Blue
        'angry': (0, 0, 255),      # Red
        'surprise': (0, 255, 255), # Yellow
        'fear': (255, 0, 255),     # Magenta
        'disgust': (255, 128, 0),  # Purple
        'neutral': (128, 128, 128) # Gray
    }
    
    # Draw rectangles and labels for each face
    for i, face_data in enumerate(face_results):
        x, y, w, h = face_data['region']
        emotion = face_data['emotion']
        confidence = face_data['confidence']
        probabilities = face_data.get('probabilities', {})
        
        # Get color for this emotion (default to white if not found)
        color = emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw rectangle with thicker line for better visibility
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 3)
        
        # Create background for text for better readability
        text_bg_color = (0, 0, 0)  # Black background
        
        # Draw emotion label with background
        label = f"{emotion.upper()} ({confidence:.2f})"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(annotated, (x, y-text_size[1]-10), (x+text_size[0]+10, y), text_bg_color, -1)
        cv2.putText(annotated, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw face number if multiple faces
        if len(face_results) > 1:
            face_label = f"Face {i+1}"
            text_size = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated, (x, y+h), (x+text_size[0]+10, y+h+text_size[1]+10), text_bg_color, -1)
            cv2.putText(annotated, face_label, (x+5, y+h+text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw emotion distribution bar if we have probabilities
        if probabilities and len(probabilities) > 0:
            # Sort emotions by probability
            sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            # Draw top 3 emotions with mini bars
            bar_width = w
            bar_height = 15
            bar_spacing = 5
            bar_y = y + h + 30  # Position below face
            
            if len(face_results) > 1:
                bar_y += 25  # Add more space if we have face numbers
            
            # Draw background for emotion bars
            cv2.rectangle(annotated, 
                         (x, bar_y), 
                         (x + bar_width, bar_y + (bar_height + bar_spacing) * min(3, len(sorted_emotions))), 
                         (0, 0, 0), 
                         -1)
            
            # Draw up to 3 emotion bars
            for j, (emo, prob) in enumerate(sorted_emotions[:3]):
                # Skip if probability is too low
                if prob < 0.05:
                    continue
                    
                emo_color = emotion_colors.get(emo, (255, 255, 255))
                emo_bar_y = bar_y + j * (bar_height + bar_spacing)
                
                # Draw emotion name
                cv2.putText(annotated, 
                           f"{emo}", 
                           (x + 5, emo_bar_y + bar_height - 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, 
                           emo_color, 
                           1)
                
                # Draw probability bar
                bar_length = int(prob * bar_width)
                cv2.rectangle(annotated, 
                             (x, emo_bar_y), 
                             (x + bar_length, emo_bar_y + bar_height), 
                             emo_color, 
                             -1)
                
                # Draw probability text
                cv2.putText(annotated, 
                           f"{prob:.2f}", 
                           (x + bar_length + 5, emo_bar_y + bar_height - 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, 
                           emo_color, 
                           1)
    
    # Add a legend for emotion colors
    legend_x = 10
    legend_y = 30
    legend_spacing = 25
    
    for i, (emotion, color) in enumerate(emotion_colors.items()):
        # Draw color box
        cv2.rectangle(annotated, 
                     (legend_x, legend_y + i * legend_spacing), 
                     (legend_x + 20, legend_y + i * legend_spacing + 20), 
                     color, 
                     -1)
        
        # Draw emotion name
        cv2.putText(annotated, 
                   emotion.upper(), 
                   (legend_x + 30, legend_y + i * legend_spacing + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   color, 
                   2)
    
    return annotated

# -------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------

def analyze_image(image_path, debug=False):
    """Analyze emotions in an image and return results with annotated image."""
    # Set environment variables
    if debug:
        os.environ['DEBUG_EMO'] = '1'
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load and process the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Get image dimensions
    h, w, c = img.shape
    
    # Analyze emotions in the image
    emotion, confidence, probs, face_count, all_face_results = analyze_emotion_ensemble(img)
    
    # Create annotated image with face boxes and emotion labels
    annotated_img = draw_face_boxes(img, all_face_results)
    
    # Add summary text at the top of the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated_img, f"Total Faces: {face_count}", (10, 30), font, 0.8, (255, 255, 255), 2)
    
    if face_count > 1:
        # Count emotions
        emotion_counts = {}
        for face in all_face_results:
            emotion_counts[face['emotion']] = emotion_counts.get(face['emotion'], 0) + 1
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        cv2.putText(annotated_img, f"Dominant Emotion: {dominant_emotion.upper()}", (10, 70), font, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(annotated_img, f"Emotion: {emotion.upper()}", (10, 70), font, 0.8, (255, 255, 255), 2)
    
    # Return comprehensive results
    results = {
        'image_dimensions': (w, h, c),
        'face_count': face_count,
        'primary_emotion': emotion,
        'primary_confidence': confidence,
        'primary_probabilities': probs,
        'all_faces': all_face_results,
        'annotated_image': annotated_img
    }
    
    # If multiple faces, add summary of emotions
    if face_count > 1:
        emotion_summary = {}
        for face in all_face_results:
            emotion_summary[face['emotion']] = emotion_summary.get(face['emotion'], 0) + 1
        results['emotion_summary'] = emotion_summary
    
    return results

# Simple usage example
if __name__ == "__main__":
    import sys
    import time
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        try:
            print(f"Analyzing image: {image_path}")
            start_time = time.time()
            
            # Analyze the image
            results = analyze_image(image_path, debug=True)
            
            # Create output directory
            os.makedirs("outputs", exist_ok=True)
            
            # Generate a descriptive filename
            if results['face_count'] > 1:
                # For multiple faces, include dominant emotion
                emotion_counts = results.get('emotion_summary', {})
                dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else results['primary_emotion']
                output_path = f"outputs/analysis-{dominant_emotion}-{results['face_count']}faces.jpg"
            else:
                # For single face
                output_path = f"outputs/analysis-{results['primary_emotion']}.jpg"
            
            # Save the annotated image
            cv2.imwrite(output_path, results['annotated_image'])
            
            # Print results
            elapsed_time = time.time() - start_time
            print(f"\nAnalysis Results (completed in {elapsed_time:.2f} seconds):")
            print(f"- Image dimensions: {results['image_dimensions'][0]}x{results['image_dimensions'][1]}")
            print(f"- Faces detected: {results['face_count']}")
            
            if results['face_count'] > 1:
                print("- Emotions detected:")
                for emotion, count in results.get('emotion_summary', {}).items():
                    print(f"  * {emotion.upper()}: {count} face(s)")
            else:
                print(f"- Emotion: {results['primary_emotion'].upper()} (confidence: {results['primary_confidence']:.2f})")
            
            print(f"- Annotated image saved to: {output_path}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Usage: python main.py <image_path>")
        print("Example: python main.py img2.jpg")
