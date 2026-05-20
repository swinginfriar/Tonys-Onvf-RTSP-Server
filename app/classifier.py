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

import threading
import urllib.request
from pathlib import Path

# Supported models. The description is shown in the UI so users can make an
# informed choice. Add to this list to expose additional models — they need
# to be ultralytics-compatible model identifiers. The `url` is the direct
# ultralytics release asset; pinned to a known version so we don't depend
# on ultralytics internals to resolve weights.
_ULTRALYTICS_ASSETS = 'https://github.com/ultralytics/assets/releases/download/v8.4.0'
MODELS = {
    'yolov8n': {
        'label': 'YOLOv8 nano',
        'file': 'yolov8n.pt',
        'url': f'{_ULTRALYTICS_ASSETS}/yolov8n.pt',
        'description': 'Fast, lower accuracy. ~50ms/frame on CPU, ~6MB model.',
        'tradeoffs': 'Best for high-volume scenes. More likely to miss small '
                     'or distant subjects. Often labels trucks/buses as "car" '
                     '— enable both if you want broad vehicle coverage.',
    },
    'yolov8s': {
        'label': 'YOLOv8 small',
        'file': 'yolov8s.pt',
        'url': f'{_ULTRALYTICS_ASSETS}/yolov8s.pt',
        'description': 'Slower, better accuracy. ~120ms/frame on CPU, ~22MB model.',
        'tradeoffs': 'Recommended for most cameras. Better at small/distant '
                     'subjects and more reliable at distinguishing truck vs '
                     'car vs bus.',
    },
    'yolo26n': {
        'label': 'YOLO26 nano (experimental)',
        'file': 'yolo26n.pt',
        'url': f'{_ULTRALYTICS_ASSETS}/yolo26n.pt',
        'description': 'Newer Ultralytics nano. ~5MB model, CPU-optimized.',
        'tradeoffs': 'Same COCO-80 classes as YOLOv8. Ultralytics claims '
                     'better speed/accuracy than YOLOv8n on CPU. Untested '
                     'in our pipeline — try on one camera first.',
    },
    'yolo26s': {
        'label': 'YOLO26 small (experimental)',
        'file': 'yolo26s.pt',
        'url': f'{_ULTRALYTICS_ASSETS}/yolo26s.pt',
        'description': 'Newer Ultralytics small. CPU-optimized successor to YOLOv8s.',
        'tradeoffs': 'Same COCO-80 classes as YOLOv8. Ultralytics claims '
                     'better speed/accuracy than YOLOv8s on CPU at a similar '
                     'compute cost. Untested in our pipeline — try on one '
                     'camera first.',
    },
}

DEFAULT_MODEL = 'yolov8n'

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
        # _inference_lock guards model.predict(); YOLO instances aren't
        # guaranteed thread-safe for concurrent inference.
        self._inference_lock = threading.Lock()
        # _load_lock guards the load/download. Separate from inference lock
        # so an in-progress inference doesn't block a parallel first-load on
        # another model, and vice versa.
        self._load_lock = threading.Lock()
        self._model = None
        self._names = None
        self.model_file = MODELS_DIR / MODELS[model_name]['file']

    def _load(self):
        # Fast path: already loaded
        if self._model is not None:
            return
        with self._load_lock:
            # Double-check after acquiring the lock — another caller may have
            # finished loading while we were blocked.
            if self._model is not None:
                return
            self._ensure_weights_file()
            # Import ultralytics here so the proxy doesn't pay the torch
            # import cost unless classification is actually enabled somewhere.
            from ultralytics import YOLO
            print(f"  [Classifier] Loading {self.model_name} from {self.model_file}")
            # Pass the absolute path so ultralytics doesn't touch the
            # process-global cwd. Loading from an absolute, existing file
            # is a pure file-read in ultralytics — no chdir, no network.
            model = YOLO(str(self.model_file))
            self._names = getattr(model, 'names', {}) or {}
            self._model = model
            print(f"  [Classifier] {self.model_name} ready ({len(self._names)} classes)")

    def _ensure_weights_file(self):
        """Download the model weights to self.model_file if not already present.

        Done explicitly (rather than letting ultralytics handle it via cwd)
        so the download lands at a deterministic absolute path with no
        impact on the process-global working directory. This avoids a
        nasty class of bug where other threads doing relative-path I/O
        during the download window write to the wrong directory.
        """
        if self.model_file.exists() and self.model_file.stat().st_size > 0:
            return
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        url = MODELS[self.model_name]['url']
        # Download to a temp file in the same directory, then atomic-rename.
        # Prevents a partially-downloaded file from being seen as "exists"
        # by a concurrent _load() call.
        tmp = self.model_file.with_suffix(self.model_file.suffix + '.part')
        print(f"  [Classifier] Downloading {self.model_name} weights to {self.model_file}")
        try:
            urllib.request.urlretrieve(url, str(tmp))
            tmp.replace(self.model_file)
        except Exception:
            # Clean up partial download so the next attempt starts fresh
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise

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
        with self._inference_lock:
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
