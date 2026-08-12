# Image-preprocess

**Deep-Learning Cell Segmentation for Fluorescent & Bright-Field Microscopy Images**

Preprocessing and segmentation of cells in **fluorescent channel** and **bright-field channel** images captured in a **16-well chip**, using a trained deep-learning segmentation model (`SegModel`).

## 📦 Contents

| File | Description |
|------|-------------|
| `1_segment.ipynb` | Cell segmentation pipeline with a deep-learning model |

## 🔬 Pipeline

The notebook segments cells in multi-channel microscopy stacks:

1. **Channel preprocessing (`seg_preprocess`)** — normalizes each channel of the input stack (`(channels, H, W)`) to a common range, handling both fluorescent and bright-field intensities;
2. **Deep-learning segmentation** — loads a pre-trained `SegModel` (e.g. `model-550-BCE.pth`) and runs inference, optionally with post-processing (binary dilation + area opening to clean the mask);
3. **Visualization** — overlays the predicted segmentation mask on the original channels for inspection.

## 🛠 Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-image](https://scikit-image.org/)

> Note: a trained model weight file (e.g. `model-550-BCE.pth`) is expected to be present in the working directory.

## 🚀 Quick Start

```python
from segment import SegModel
import torch

# Load model
segment = SegModel("model-550-BCE.pth", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Segment a frame
im = seg_preprocess(frames[j])
pred = segment.predict(im)
```

## 📄 License

For educational and research purposes.
