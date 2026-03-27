# 🧠 NeuroScan AI — Tumor Detection Web App

A Flask-based deployment for your `.pth` deep learning model with a clinical-grade radiologist interface.  
Grad-CAM overlays highlight suspected tumor regions directly on the scan.

---

## Project Structure

```
tumor_detector/
├── app.py                  # Flask backend (inference + Grad-CAM)
├── requirements.txt        # Python dependencies
├── model.pth               # ← PUT YOUR MODEL HERE
└── static/
    └── index.html          # Radiologist frontend UI
```

---

## Quick Start

### 1. Place your model
Copy your trained model file into this folder and name it `model.pth`  
(or set the env variable `MODEL_PATH` to the actual path).

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure for your model (edit top of `app.py`)
```python
MODEL_PATH  = "model.pth"       # path to your .pth file
NUM_CLASSES = 2                  # number of output classes
CLASS_NAMES = "No Tumor,Tumor"  # comma-separated, class 0 = no tumor
IMG_SIZE    = 224                # input image size your model expects
```

### 4. Run
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Environment Variables (alternative to editing app.py)
| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model.pth` | Path to `.pth` file |
| `NUM_CLASSES` | `2` | Number of output classes |
| `CLASS_NAMES` | `No Tumor,Tumor` | Comma-separated class names |
| `IMG_SIZE` | `224` | Input resolution |

Example:
```bash
MODEL_PATH=weights/brain_tumor_resnet50.pth NUM_CLASSES=4 \
CLASS_NAMES="Glioma,Meningioma,No Tumor,Pituitary" python app.py
```

---

## Adapting to a Different Architecture

The default backbone is **ResNet-50**. If you trained a different model, edit `build_model()` in `app.py`:

```python
# Example: EfficientNet-B0
from torchvision.models import efficientnet_b0
def build_model():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model
```

Also update the `GradCAM` target layer:
```python
# For EfficientNet
GRAD_CAM = GradCAM(MODEL, target_layer_name="features")
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the radiologist UI |
| `/predict` | POST | Accepts `multipart/form-data` with `image` field; returns JSON |
| `/health` | GET | Model status and device info |

### `/predict` Response Example
```json
{
  "prediction": "Tumor",
  "predicted_idx": 1,
  "probabilities": {
    "No Tumor": 3.41,
    "Tumor": 96.59
  },
  "is_tumor": true,
  "original_image": "<base64 PNG>",
  "overlay_image": "<base64 PNG with Grad-CAM>"
}
```

---

## Notes
- Grad-CAM targets `layer4` (ResNet's deepest conv block) by default — produces the best localization.
- The disclaimer in the UI is mandatory: this is a decision-support tool, not a diagnostic device.
- For production deployment, add HTTPS (via nginx + certbot) and authentication.
