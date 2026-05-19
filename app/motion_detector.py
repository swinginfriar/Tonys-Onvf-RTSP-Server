"""Per-camera motion detection worker.

Reads frames from the MediaMTX local substream via a small ffmpeg subprocess
that outputs downscaled grayscale rawvideo to a pipe, applies MOG2 background
subtraction, and feeds boolean observations into the shared MotionController.
The controller handles debounce + ONVIF event emission, so the worker only
reports raw frame-level observations.

Lazy-imports cv2 and numpy so cameras without motion enabled don't need
OpenCV installed.
"""

import subprocess
import threading
import time
from urllib.parse import quote

from .ffmpeg_manager import FFmpegManager
from .motion_controller import get_motion_controller

# Hardcoded internals — could be exposed in later PRs
SCALE_WIDTH = 640                # downscale frames to this width for analysis
MOG2_VAR_THRESHOLD = 16          # default MOG2 sensitivity
MOG2_HISTORY = 500               # background model history (frames)

# Reconnect backoff
BACKOFF_INITIAL_S = 2
BACKOFF_MAX_S = 30

# Defaults if not in per-camera config
DEFAULT_FPS = 3
DEFAULT_MIN_AREA_PERCENT = 1.0
DEFAULT_MIN_MOTION_FRAMES = 2


def _import_cv():
    """Lazy import cv2 + numpy. Raises ImportError with a clear message if missing."""
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
        return cv2, np
    except ImportError as e:
        raise ImportError(
            "OpenCV (opencv-python-headless) and numpy are required for motion detection. "
            "Install with: pip install opencv-python-headless numpy"
        ) from e


class MotionWorker:
    """Background thread that detects motion on one camera's substream."""

    def __init__(self, camera):
        self.camera = camera
        self.cfg = dict(getattr(camera, 'motion', None) or {})
        self._stop_event = threading.Event()
        self._thread = None
        self._ffmpeg_proc = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"motion-{self.camera.id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout=5):
        self._stop_event.set()
        self._terminate_ffmpeg()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self):
        try:
            cv2, np = _import_cv()
        except ImportError as e:
            print(f"  [Motion] {self.camera.name}: cannot start — {e}")
            return

        print(f"  [Motion] {self.camera.name}: worker started")
        backoff = BACKOFF_INITIAL_S
        while not self._stop_event.is_set():
            try:
                self._detect_loop(cv2, np)
                # If _detect_loop returns cleanly (only happens on stop), exit
                break
            except Exception as e:
                if self._stop_event.is_set():
                    break
                print(f"  [Motion] {self.camera.name}: stream error ({e}), reconnecting in {backoff}s")
                self._terminate_ffmpeg()
                # Sleep responsively so stop() returns quickly
                if self._stop_event.wait(timeout=backoff):
                    break
                backoff = min(backoff * 2, BACKOFF_MAX_S)
        print(f"  [Motion] {self.camera.name}: worker stopped")

    def _detect_loop(self, cv2, np):
        scale_w, scale_h = self._compute_scale()
        frame_bytes = scale_w * scale_h
        fps = max(1, int(self.cfg.get('fps', DEFAULT_FPS)))
        min_area_percent = float(self.cfg.get('min_area_percent', DEFAULT_MIN_AREA_PERCENT))
        min_motion_frames = max(1, int(self.cfg.get('min_motion_frames', DEFAULT_MIN_MOTION_FRAMES)))

        # Build the zone mask once. None = full frame (today's behavior).
        zone_mask = self._build_zone_mask(cv2, np, scale_w, scale_h)
        # Threshold is % of the zone area (or full frame if no zones), so a 1%
        # setting means the same thing regardless of how the user has scoped.
        zone_area_px = int(zone_mask.sum()) if zone_mask is not None else scale_w * scale_h
        motion_threshold_px = max(1, int(min_area_percent * 0.01 * zone_area_px))

        url = self._build_stream_url()
        cmd = self._build_ffmpeg_cmd(url, scale_w, scale_h, fps)
        self._ffmpeg_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        bg = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=False,
        )
        controller = get_motion_controller()
        consecutive = 0

        while not self._stop_event.is_set():
            buf = self._read_exact(self._ffmpeg_proc.stdout, frame_bytes)
            if buf is None:
                raise EOFError("ffmpeg pipe closed")
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(scale_h, scale_w)
            fg = bg.apply(frame)
            if zone_mask is not None:
                fg = fg * zone_mask
            motion_area = int(cv2.countNonZero(fg))

            if motion_area >= motion_threshold_px:
                consecutive += 1
            else:
                consecutive = 0

            is_motion = consecutive >= min_motion_frames
            controller.set_motion(self.camera, is_motion, source='detector')

    def _build_zone_mask(self, cv2, np, scale_w, scale_h):
        """Build a uint8 binary mask (0/1) from configured zones.

        Returns None when no zones are configured — in that case the caller
        treats the full frame as the analysis region (today's behavior).

        Each zone: {name, enabled, exclude?, polygon: [[x,y], ...]} with
        coords normalized to 0.0-1.0 so they scale across resolutions.
        Final mask = (OR of enabled include-zones) AND NOT (OR of enabled
        exclude-zones). If no include zones are defined, the full frame is
        treated as the include region so users can add only exclude-zones
        to mask out wind/tree areas without re-declaring the rest.
        """
        zones = self.cfg.get('zones') or []
        if not zones:
            return None

        include_mask = np.zeros((scale_h, scale_w), dtype=np.uint8)
        exclude_mask = np.zeros((scale_h, scale_w), dtype=np.uint8)
        has_include = False

        for z in zones:
            if not z.get('enabled', True):
                continue
            polygon = z.get('polygon') or []
            if len(polygon) < 3:
                continue
            try:
                pts = np.array(
                    [[int(round(float(p[0]) * scale_w)),
                      int(round(float(p[1]) * scale_h))] for p in polygon],
                    dtype=np.int32,
                )
            except (TypeError, ValueError, IndexError) as e:
                print(f"  [Motion] {self.camera.name}: skipping malformed zone {z.get('name','?')}: {e}")
                continue
            if z.get('exclude', False):
                cv2.fillPoly(exclude_mask, [pts], 1)
            else:
                cv2.fillPoly(include_mask, [pts], 1)
                has_include = True

        if not has_include:
            include_mask[:] = 1

        mask = include_mask * (1 - exclude_mask)
        # Tell operators which zone area we ended up with — useful when
        # tuning min_area_percent against a small zone.
        zone_pct = float(mask.sum()) * 100.0 / (scale_w * scale_h)
        print(f"  [Motion] {self.camera.name}: zone mask covers {zone_pct:.1f}% of frame")
        return mask

    def _read_exact(self, stream, n):
        """Read exactly n bytes from stream. Return None on EOF/short read."""
        buf = bytearray()
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _compute_scale(self):
        src_w = max(1, int(getattr(self.camera, 'sub_width', 640) or 640))
        src_h = max(1, int(getattr(self.camera, 'sub_height', 480) or 480))
        # If substream is disabled, fall back to main dims (we'll still read sub URL
        # if MediaMTX has main aliased, but the dims should still match the source)
        scale_w = SCALE_WIDTH
        scale_h = max(2, int(src_h * scale_w / src_w))
        return scale_w, scale_h

    def _build_stream_url(self):
        """Construct the MediaMTX local substream URL."""
        manager = getattr(self.camera, 'manager', None)
        rtsp_port = getattr(manager, 'rtsp_port', 8554) if manager else 8554
        # Use sub-stream if available, else main
        suffix = "_sub" if not getattr(self.camera, 'disable_substream', False) else "_main"
        path = f"{self.camera.path_name}{suffix}"

        if manager and getattr(manager, 'rtsp_auth_enabled', False):
            user = quote(getattr(manager, 'global_username', 'admin') or 'admin', safe='')
            pwd = quote(getattr(manager, 'global_password', 'admin') or 'admin', safe='')
            return f"rtsp://{user}:{pwd}@127.0.0.1:{rtsp_port}/{path}"
        return f"rtsp://127.0.0.1:{rtsp_port}/{path}"

    def _build_ffmpeg_cmd(self, url, scale_w, scale_h, fps):
        ffmpeg = FFmpegManager().get_ffmpeg_path()
        return [
            ffmpeg,
            '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-rtsp_transport', 'tcp',
            '-i', url,
            '-vf', f'fps={fps},scale={scale_w}:{scale_h}',
            '-pix_fmt', 'gray',
            '-an', '-sn',
            '-f', 'rawvideo',
            'pipe:1',
        ]

    def _terminate_ffmpeg(self):
        proc = self._ffmpeg_proc
        self._ffmpeg_proc = None
        if not proc:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
        except Exception:
            pass
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass


# Per-camera worker registry so manager can stop them on shutdown.
_WORKERS = {}
_WORKERS_LOCK = threading.Lock()


def start_worker(camera):
    """Reconcile the MotionWorker to the camera's current motion config.

    Handles all four transitions cleanly: stops any existing worker first,
    then starts a fresh one only if motion is currently enabled. Called both
    from camera lifecycle (camera.start) and from the config API (so toggling
    motion off-or-on at runtime takes effect without restarting the camera).
    """
    with _WORKERS_LOCK:
        existing = _WORKERS.pop(camera.id, None)
    if existing:
        existing.stop()
        # Clear any in-flight motion state so we don't leave a stuck-ON event
        # behind when the user toggles motion off or reconfigures it.
        get_motion_controller().set_motion(camera, False, source='worker_stop', immediate=True)

    cfg = getattr(camera, 'motion', None) or {}
    if not cfg.get('enabled', False):
        return None

    worker = MotionWorker(camera)
    with _WORKERS_LOCK:
        _WORKERS[camera.id] = worker
    worker.start()
    return worker


def stop_worker(camera):
    """Stop a camera's MotionWorker if running and clear any in-flight motion state."""
    with _WORKERS_LOCK:
        worker = _WORKERS.pop(camera.id, None)
    if worker:
        worker.stop()
        # Clear any stuck-ON state so the NVR doesn't see motion forever
        get_motion_controller().set_motion(camera, False, source='worker_stop', immediate=True)
