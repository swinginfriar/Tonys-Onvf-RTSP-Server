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
        # BGR24 = 3 bytes per pixel. We keep the color frame around so the
        # classifier (if enabled) can use it; MOG2 still runs on a grayscale
        # conversion of the same frame.
        frame_bytes = scale_w * scale_h * 3
        fps = max(1, int(self.cfg.get('fps', DEFAULT_FPS)))
        min_area_percent = float(self.cfg.get('min_area_percent', DEFAULT_MIN_AREA_PERCENT))
        min_motion_frames = max(1, int(self.cfg.get('min_motion_frames', DEFAULT_MIN_MOTION_FRAMES)))

        # Resolve classification settings (off by default).
        cls_cfg = (self.cfg.get('classification') or {})
        cls_enabled = bool(cls_cfg.get('enabled', False))
        cls_model = cls_cfg.get('model') or 'yolov8s'
        cls_min_conf = float(cls_cfg.get('min_confidence', 0.4))
        cls_classes = list(cls_cfg.get('classes') or [])
        # 'sub' (default): classify the existing 640px sub-stream frame already
        #   in memory. Fast (~0ms extra capture cost). Bad at distant/small subjects.
        # 'main': capture a fresh full-resolution main-stream frame when
        #   classification fires. Adds ~500-1000ms latency per event but
        #   the higher resolution dramatically improves recall on distant subjects.
        cls_stream = (cls_cfg.get('stream') or 'sub').lower()
        if cls_stream not in ('sub', 'main'):
            cls_stream = 'sub'
        classifier = None
        if cls_enabled:
            try:
                from .classifier import get_classifier
                classifier = get_classifier(cls_model)
                print(f"  [Motion] {self.camera.name}: classification ON ({cls_model}, "
                      f"stream={cls_stream}, min_conf={cls_min_conf}, "
                      f"classes={cls_classes or 'ALL'})")
            except Exception as e:
                print(f"  [Motion] {self.camera.name}: classifier failed to load — disabling: {e}")
                classifier = None

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
        # When classification is on, we only fire an ONVIF motion event after
        # one frame of a motion event has been classified as a wanted class.
        # Once fired, we stop re-classifying for this event until motion stops.
        classified_for_event = False
        # Event-hold: after firing motion=true, we hold that state for at
        # least N seconds even if scene motion stops earlier. This makes the
        # marker on the NVR's timeline long enough to actually click on.
        # Duration is from camera.motion.min_event_duration_ms, scaled up by
        # classification confidence (higher confidence -> longer hold).
        base_hold_s = max(0.0, int(self.cfg.get('min_event_duration_ms', 5000))) / 1000.0
        event_min_end_time = 0.0

        while not self._stop_event.is_set():
            buf = self._read_exact(self._ffmpeg_proc.stdout, frame_bytes)
            if buf is None:
                raise EOFError("ffmpeg pipe closed")
            frame_bgr = np.frombuffer(buf, dtype=np.uint8).reshape(scale_h, scale_w, 3)
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            fg = bg.apply(frame_gray)
            if zone_mask is not None:
                fg = fg * zone_mask
            motion_area = int(cv2.countNonZero(fg))

            if motion_area >= motion_threshold_px:
                consecutive += 1
            else:
                consecutive = 0

            raw_motion = consecutive >= min_motion_frames
            now = time.time()

            if not raw_motion:
                # Scene motion has stopped — but if we're still inside the
                # min-event-duration window, keep holding motion=TRUE so the
                # NVR timeline marker stays a meaningful length.
                if event_min_end_time > 0 and now < event_min_end_time:
                    controller.set_motion(self.camera, True, source='detector:hold')
                    continue
                # Past the hold window — fire OFF (idempotent) and reset state.
                controller.set_motion(self.camera, False, source='detector')
                classified_for_event = False
                event_min_end_time = 0.0
                continue

            # raw_motion == True
            if classifier is None:
                # No classification — fire normally with the base hold duration.
                controller.set_motion(self.camera, True, source='detector')
                if event_min_end_time == 0.0:
                    event_min_end_time = now + base_hold_s
                continue

            if classified_for_event:
                # Already fired for this event; keep state TRUE (idempotent).
                controller.set_motion(self.camera, True, source='detector')
                continue

            # First motion frame for this event with classification on — classify.
            # If the user opted into the main stream for classification, pull a
            # fresh high-res frame; otherwise classify the sub-stream frame we
            # already have in memory.
            try:
                if cls_stream == 'main':
                    cls_frame = self._capture_main_frame(cv2)
                    if cls_frame is None:
                        # Capture failed — fall back to sub frame so we don't lose the event
                        cls_frame = frame_bgr
                else:
                    cls_frame = frame_bgr
                passed, top_class, dets = classifier.classify(
                    cls_frame, cls_classes, cls_min_conf,
                )
            except Exception as e:
                print(f"  [Motion] {self.camera.name}: classifier error: {e}")
                passed, top_class, dets = False, None, []

            if passed:
                # Scale the hold duration by classification confidence so a
                # high-confidence "definitely a person" event lasts longer on
                # the timeline than a marginal one. Bands chosen empirically:
                #   conf >= 0.80 -> 2.5x base
                #   conf >= 0.65 -> 1.5x base
                #   else         -> 1.0x base
                top_conf = float(dets[0]['conf']) if dets else 0.0
                if top_conf >= 0.80:
                    mult = 2.5
                elif top_conf >= 0.65:
                    mult = 1.5
                else:
                    mult = 1.0
                hold_s = base_hold_s * mult
                event_min_end_time = now + hold_s
                print(f"  [Motion] {self.camera.name}: classified as '{top_class}' "
                      f"(conf={top_conf:.2f}) — firing event (hold {hold_s:.1f}s)")
                controller.set_motion(self.camera, True, source=f'detector:{top_class}')
                classified_for_event = True
            # If classification failed, do NOT fire. Next motion frame will
            # re-attempt — important for subjects that walk into view partway.

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

    def _build_stream_url(self, stream='sub'):
        """Construct a MediaMTX local stream URL for this camera.

        stream='sub' (default) maps to <path>_sub; 'main' maps to <path>_main.
        If the camera has the sub-stream disabled, both fall back to main
        (since that's the only stream available).
        """
        manager = getattr(self.camera, 'manager', None)
        rtsp_port = getattr(manager, 'rtsp_port', 8554) if manager else 8554
        if getattr(self.camera, 'disable_substream', False):
            suffix = '_main'
        else:
            suffix = '_main' if stream == 'main' else '_sub'
        path = f"{self.camera.path_name}{suffix}"

        if manager and getattr(manager, 'rtsp_auth_enabled', False):
            user = quote(getattr(manager, 'global_username', 'admin') or 'admin', safe='')
            pwd = quote(getattr(manager, 'global_password', 'admin') or 'admin', safe='')
            return f"rtsp://{user}:{pwd}@127.0.0.1:{rtsp_port}/{path}"
        return f"rtsp://127.0.0.1:{rtsp_port}/{path}"

    def _capture_main_frame(self, cv2):
        """Grab a single full-resolution main-stream frame for classification.

        Uses the same FFmpegManager.capture_snapshot path as /onvif/snapshot —
        cold ffmpeg start, ~500-1000ms typical latency. Worth it for distant
        subjects since main resolution gives the model ~9x the pixel density.
        Returns a BGR numpy array, or None on failure.
        """
        import os, tempfile
        from .ffmpeg_manager import FFmpegManager
        url = self._build_stream_url(stream='main')
        fd, path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        try:
            ok, _err = FFmpegManager().capture_snapshot(url, path, timeout=8)
            if not ok:
                return None
            return cv2.imread(path)
        finally:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

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
