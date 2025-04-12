import os
import base64
import io

import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

VARIANT_PATHS = {
    "waterIconCup": "models/waterIconCup.onnx",
}

onnx_models = {}
model_info = {}

def load_onnx_model(variant):
    if variant in onnx_models:
        return onnx_models[variant]
        
    model_path = VARIANT_PATHS.get(variant)
    if not model_path:
        raise ValueError(f"Unknown variant: {variant}")
    
    session = onnxruntime.InferenceSession(model_path)
    
    input_details = session.get_inputs()
    channels = input_details[0].shape[1] if len(input_details[0].shape) > 1 else 1
    
    onnx_models[variant] = session
    model_info[variant] = {"channels": channels}
    
    print(f"Model for {variant} expects {channels} channel(s)")
    return session

def process_image(image_data, variant):
    try:
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        session = load_onnx_model(variant)
        channels = model_info[variant]["channels"]
        
        if channels == 1:
            img = img.convert("L")
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            img = img.convert("RGB")

            transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.ToTensor(),
            ])
        
        image1 = img.crop((0, 200, 200, 400))
        image1_transformed = transform(image1)
        
        max_similarity = float("-inf")
        max_index = -1
        
        for i in range(img.width // 200):
            image2 = img.crop((i * 200, 0, (i + 1) * 200, 200))
            image2_transformed = transform(image2)
            
            image1_np = image1_transformed.unsqueeze(0).numpy()
            image2_np = image2_transformed.unsqueeze(0).numpy()
            
            output = session.run(None, {
                    "input_left": image1_np, 
                    "input_right": image2_np
            })[0]
            
            similarity = output.item()
            
            if similarity > max_similarity:
                max_similarity = similarity
                max_index = i
        
        return {
            "success": True,
            "result": {
                "best_match_index": max_index,
                "similarity_score": max_similarity / 100
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.route('/match_image', methods=['POST'])
def match_image():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        image_data = data.get('image')
        variant = data.get('variant')
        
        if not image_data:
            return jsonify({"success": False, "error": "No image provided"}), 400
        if not variant:
            return jsonify({"success": False, "error": "No variant provided"}), 400
        
        result = process_image(image_data, variant)
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            "success": False, 
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    for variant in VARIANT_PATHS:
        try:
            load_onnx_model(variant)
            print(f"Loaded ONNX model for variant: {variant}")
        except Exception as e:
            print(f"Failed to load ONNX model for variant {variant}: {e}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)