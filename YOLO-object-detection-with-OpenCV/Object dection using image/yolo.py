import numpy as np
import cv2
from PIL import Image
import gradio as gr
from collections import defaultdict

# Load COCO class labels
LABELS = open('yolo-coco/coco.names').read().strip().split("\n")

# Color settings (Red for bounding boxes, White for text)
RED_COLOR = (0, 0, 255)
WHITE_COLOR = (255, 255, 255)

# Load YOLO model weights and config
weightsPath = 'yolo-coco/yolov3.weights'
configPath = 'yolo-coco/yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Advanced image enhancement with stronger sharpening
def enhance_image(image):
    # Convert to LAB for better luminance processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge LAB and convert back to BGR
    enhanced_image = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

    # Multi-step sharpening: Gaussian Blur + Weighted Add
    blur = cv2.GaussianBlur(enhanced_image, (3, 3), 2)
    sharpened = cv2.addWeighted(enhanced_image, 2.5, blur, -1.5, 0)

    # Contrast stretching to boost pixel intensity
    stretched = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

    return stretched

# Enhanced object detection function
def detect_objects(image):
    image = enhance_image(image)  # Apply enhancement
    (H, W) = image.shape[:2]

    # Resize if the image is too large
    if max(H, W) > 1280:
        scale = 1280 / max(H, W)
        image = cv2.resize(image, (int(W * scale), int(H * scale)))
        (H, W) = image.shape[:2]

    # Convert to RGB for PIL compatibility
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a blob for YOLO
    blob = cv2.dnn.blobFromImage(image_rgb, 1 / 255.0, (832, 832), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists to store detections
    boxes, confidences, classIDs = [], [], []
    object_counts = defaultdict(int)

    # Process YOLO outputs
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.35:  # Detection threshold
                # Calculate coordinates
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = max(0, int(centerX - (width / 2)))
                y = max(0, int(centerY - (height / 2)))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply Non-Maximum Suppression (NMS)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.2, nms_threshold=0.45)

    # Draw bounding boxes and display labels
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw red bounding box
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), RED_COLOR, 2)

            # Format the label with white text and percentage confidence
            label = f"{LABELS[classIDs[i]]}: {confidences[i] * 100:.2f}%"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            text_x = x
            text_y = y - 10 if y - 10 > 10 else y + text_height + 10

            # Draw background rectangle for text
            cv2.rectangle(
                image_rgb,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                (0, 0, 0), 
                cv2.FILLED
            )

            # Display the label in white
            cv2.putText(
                image_rgb, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE_COLOR, 2
            )

            # Track object counts
            object_counts[LABELS[classIDs[i]]] += 1

    # Convert to PIL image for Gradio output
    output_image = Image.fromarray(image_rgb)

    # Prepare summary of detected objects
    object_list = "\n".join([f"{obj}: {count}" for obj, count in object_counts.items()])

    return output_image, object_list

# Gradio UI setup
with gr.Blocks(title="YOLO Object Detection") as demo:
    gr.Markdown("# YOLO Object Detection App")

    with gr.Row():
        input_image = gr.Image(label="Upload Image", type="numpy")
        output_image = gr.Image(label="Detected Objects", type="pil")

    output_text = gr.Textbox(label="Object Counts", lines=10)

    detect_button = gr.Button("Detect")
    detect_button.click(fn=detect_objects, inputs=input_image, outputs=[output_image, output_text])

# Launch Gradio app
demo.launch(share=True)

