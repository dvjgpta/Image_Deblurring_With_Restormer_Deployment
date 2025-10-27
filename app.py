import cv2
import torch
import numpy as np
import torch.nn.functional as F
from flask import Flask, request, render_template, jsonify, send_file
from io import BytesIO
from archs.restormer_arch import Restormer

app = Flask(__name__)

# ---- Load model ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Restormer()
checkpoint = torch.load(
    '/mnt/DATA/EE22B013/Btech_project/Model/experiments/Restormer_GOPRO/models/restormer_latest.pth',
    map_location=device
)

if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()


def deblur_image(img_path):
    factor = 8
    patch_size = 512
    stride = patch_size

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    h, w = img.shape[2], img.shape[3]
    H, W = ((h + factor) // factor * factor, (w + factor) // factor * factor)
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    padding = (0, padw, 0, padh)
    img = F.pad(img, padding, 'reflect')

    out = torch.zeros_like(img)
    with torch.no_grad():
        for i in range(0, img.shape[2], stride):
            for j in range(0, img.shape[3], stride):
                patch = img[:, :, i:i + patch_size, j:j + patch_size]
                patch_out = model(patch)
                out[:, :, i:i + patch_out.shape[2], j:j + patch_out.shape[3]] = patch_out
                torch.cuda.empty_cache()

    out = out[:, :, :h, :w]
    out = torch.clamp(out, 0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
    out = (out * 255).astype(np.uint8)
    return out


@app.route('/')
def home():
    return render_template('index.html')


# @app.route('/infer', methods=['POST'])
# def infer_image():
#     file = request.files['image']
#     input_path = '/tmp/input.png'
#     file.save(input_path)

#     result = deblur_image(input_path)
#     _, buffer = cv2.imencode('.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    # return send_file(BytesIO(buffer), mimetype='image/png')
import base64

@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected image'})

        input_path = '/tmp/input.png'
        file.save(input_path)

        result = deblur_image(input_path)

        _, buffer = cv2.imencode('.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'result': f"data:image/png;base64,{img_str}"})

    except Exception as e:
        print("Error during inference:", str(e))
        return jsonify({'error': 'Something went wrong'})



if __name__ == '__main__':
    app.run(debug=True)
