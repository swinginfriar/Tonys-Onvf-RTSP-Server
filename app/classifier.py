"""YOLO-based object classification for the motion detection pipeline.

Lazy-loaded so cameras without classification enabled never import torch
or load any models. When a camera turns classification on, the first call
into get_classifier() imports ultralytics, downloads the model weights if
needed, and caches the instance for sharing across other cameras using the
same model.

Typical usage from MotionWorker:

    from .classifier import get_classifier
    c = get_classifier('yolov8s')
    passed, top_class, dets = c.classify(frame_bgr, ['person', 'car'], 0.4)
    if passed:
        controller.set_motion(camera, True, source='detector')

Models are cached on disk under <repo>/models/yolo/. First-use download takes
1-2s for nano, 2-4s for small. After that, loads are essentially instant.
"""

import os
import threading
from pathlib import Path

# Supported models. The description is shown in the UI so users can make an
# informed choice. Add to this list to expose additional models — they need
# to be ultralytics-compatible model identifiers.
MODELS = {
    'yolov8n': {
        'label': 'YOLOv8 nano',
        'file': 'yolov8n.pt',
        'description': 'Fast, lower accuracy. ~50ms/frame on CPU, ~6MB model.',
        'tradeoffs': 'Best for high-volume scenes. More likely to miss small '
                     'or distant subjects. Often labels trucks/buses as "car" '
                     '— enable both if you want broad vehicle coverage.',
    },
    'yolov8s': {
        'label': 'YOLOv8 small',
        'file': 'yolov8s.pt',
        'description': 'Slower, better accuracy. ~120ms/frame on CPU, ~22MB model.',
        'tradeoffs': 'Recommended for most cameras. Better at small/distant '
                     'subjects and more reliable at distinguishing truck vs '
                     'car vs bus.',
    },
}

DEFAULT_MODEL = 'yolov8s'

# COCO class names that the YOLOv8 models can detect, grouped for the UI.
# The keys are display group names; values are class names from COCO-80.
COCO_CLASS_GROUPS = {
    'People': ['person'],
    'Vehicles': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'boat', 'airplane'],
    'Animals': ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    'Other': ['backpack', 'umbrella', 'handbag', 'suitcase', 'sports ball'],
}

# Default classes a fresh user probably wants — security-relevant subjects
DEFAULT_CLASSES = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']

# Where to store downloaded model weights. Kept inside the repo dir so they
# travel with the install (and so they're easy to find).
MODELS_DIR = Path(__file__).resolve().parent.parent / 'models' / 'yolo'


def is_available():
    """True if the ultralytics package is installed and importable.

    Used by the API + UI to surface a friendly 'install ultralytics' message
    before the user enables classification, rather than failing silently
    when the first motion event tries to classify.
    """
    try:
        import importlib.util
        return importlib.util.find_spec('ultralytics') is not None
    except Exception:
        return False


def list_models():
    """Return UI-facing metadata for all supported models."""
    return [
        {'name': name, **{k: v for k, v in info.items() if k != 'file'}}
        for name, info in MODELS.items()
    ]


def list_classes():
    """Return the grouped class catalog for UI checkboxes."""
    return COCO_CLASS_GROUPS


class _Classifier:
    """Wraps one YOLO model instance. Loaded once per process per model."""

    def __init__(self, model_name):
        if model_name not in MODELS:
            raise ValueError(f"Unknown model {model_name!r}; supported: {list(MODELS)}")
        self.model_name = model_name
        self._lock = threading.Lock()  # YOLO inference isn't always thread-safe
        self._model = None  # Loaded on first classify()
        self._names = None  # int->str class name mapping
        self.model_file = MODELS_DIR / MODELS[model_name]['file']

    def _load(self):
        if self._model is not None:
            return
        # Import ultralytics here so users without classification enabled
        # never pay the import cost (which pulls in torch + a lot more).
        from ultralytics import YOLO
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # Ultralytics resolves weights either from a local path or by
        # downloading from its release server. Passing the file name lets it
        # download into the current working directory by default — we cd
        # there briefly so the file lands in MODELS_DIR.
        cwd = os.getcwd()
        try:
            os.chdir(MODELS_DIR)
            print(f"  [Classifier] Loading {self.model_name} (downloads on first use)...")
            self._model = YOLO(MODELS[self.model_name]['file'])
            # Cache the class-id -> name dict for quick lookup
            self._names = self._model.names if hasattr(self._model, 'names') else {}
            print(f"  [Classifier] {self.model_name} ready ({len(self._names)} classes)")
        finally:
            os.chdir(cwd)

    def classify(self, image_bgr, classes_filter=None, min_confidence=0.4):
        """Run inference on one BGR frame.

        Returns (passed, top_class, detections):
          passed         - True if at least one detection matches the filter
          top_class      - name of the highest-confidence matching class (or None)
          detections     - list of dicts [{class, conf, bbox}, ...] of all
                           matching detections (above min_confidence and in
                           classes_filter if provided)
        """
        self._load()
        filter_set = set(classes_filter) if classes_filter else None
        with self._lock:
            results = self._model.predict(
                image_bgr,
                verbose=False,
                imgsz=640,
                conf=float(min_confidence),
            )
        if not results:
            return False, None, []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return False, None, []

        matches = []
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            conf = float(r.boxes.conf[i].item())
            cls_name = self._names.get(cls_id, str(cls_id))
            if filter_set and cls_name not in filter_set:
                continue
            xyxy = [float(v) for v in r.boxes.xyxy[i].tolist()]
            matches.append({'class': cls_name, 'conf': conf, 'bbox': xyxy})

        if not matches:
            return False, None, []
        matches.sort(key=lambda m: -m['conf'])
        return True, matches[0]['class'], matches


# Process-wide registry so cameras using the same model share one instance
_REGISTRY = {}
_REGISTRY_LOCK = threading.Lock()


def get_classifier(model_name=DEFAULT_MODEL):
    """Return the shared _Classifier for this model, loading on first call."""
    with _REGISTRY_LOCK:
        c = _REGISTRY.get(model_name)
        if c is None:
            c = _Classifier(model_name)
            _REGISTRY[model_name] = c
    return c
