import io
import os
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

app = Flask(__name__, static_folder="static")
CORS(app)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "model.pth"
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── MODEL LOADING ───────────────────────────────────────────────────────────
def build_model():
    # FIX 1: EfficientNet-B0 (your .pth has "module.features" keys, not ResNet keys)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


def load_model():
    model = build_model()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    # FIX 2: Strip "module." prefix added by DataParallel during training
    new_state = {}
    for k, v in state.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        new_state[new_key] = v

    model.load_state_dict(new_state)
    model.to(DEVICE)
    model.eval()
    print(f"[✓] Model loaded from {MODEL_PATH}  |  device: {DEVICE}")
    return model


MODEL = load_model()

# ─── TRANSFORMS ──────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── GRAD-CAM ────────────────────────────────────────────────────────────────
class GradCAM:
    # FIX 3: Target EfficientNet's last conv block (features[-1]), not "layer4"
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hook()

    def _hook(self):
        target = self.model.features[-1]

        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        target.register_forward_hook(fwd_hook)
        target.register_full_backward_hook(bwd_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam)
        cam     = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), class_idx, F.softmax(output, dim=1)[0].cpu().detach().numpy()


GRAD_CAM = GradCAM(MODEL)

# ─── HELPER: overlay heatmap ─────────────────────────────────────────────────
def overlay_heatmap(pil_image: Image.Image, cam: np.ndarray, alpha=0.45) -> Image.Image:
    img_np = np.array(pil_image.convert("RGB"))
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img_np)
    return Image.fromarray(overlay)


def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    cam, pred_idx, probs = GRAD_CAM.generate(tensor)

    overlay_img = overlay_heatmap(pil_img, cam)

    def resize_for_display(img, max_w=512):
        w, h = img.size
        if w > max_w:
            ratio = max_w / w
            img = img.resize((max_w, int(h * ratio)), Image.LANCZOS)
        return img

    orig_b64    = pil_to_b64(resize_for_display(pil_img))
    overlay_b64 = pil_to_b64(resize_for_display(overlay_img))

    response = {
        "prediction":    CLASS_NAMES[pred_idx],
        "predicted_idx": int(pred_idx),
        "probabilities": {CLASS_NAMES[i]: float(round(probs[i] * 100, 2))
                          for i in range(NUM_CLASSES)},
        "original_image": orig_b64,
        "overlay_image": overlay_b64,
        "is_tumor":       bool(pred_idx != 3),   # notumor is index 2
    }
    return jsonify(response)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE),
                    "model": MODEL_PATH, "classes": CLASS_NAMES})


if __name__ == "__main__":
    port = int(os.environ.get("PORT",10000 ))
    app.run(host="0.0.0.0", port=port, debug=False)
     