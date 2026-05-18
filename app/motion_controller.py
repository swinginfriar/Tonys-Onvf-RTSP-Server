"""Motion state controller — debounces input and emits ONVIF motion events.

Sources of motion (synthetic endpoints, real detection workers, etc.) call
set_motion(camera_id, is_motion). The controller debounces flap, tracks
on/off state per camera, and publishes the proper ONVIF topics to the event
bus when state transitions happen.

Two topics are emitted on every transition for broad NVR compatibility:
  - tns1:RuleEngine/CellMotionDetector/Motion  (IsMotion=true|false)
  - tns1:VideoSource/MotionAlarm               (State=true|false)
"""

import threading
import time

from .event_bus import get_event_bus

TOPIC_CELL_MOTION = "tns1:RuleEngine/CellMotionDetector/Motion"
TOPIC_MOTION_ALARM = "tns1:VideoSource/MotionAlarm"

# Defaults if the camera has no motion config. Off-delay is long enough to bridge
# typical detection gaps; on-delay is short so triggers feel responsive.
DEFAULT_ON_DELAY_MS = 500
DEFAULT_OFF_DELAY_MS = 3000


class MotionController:
    def __init__(self):
        self._lock = threading.RLock()
        # camera_id -> {"state": bool, "pending": bool|None, "pending_since": float}
        self._states = {}

    def _get_delays(self, camera):
        """Read on/off debounce delays from the camera's motion config, with defaults."""
        cfg = getattr(camera, 'motion', None) or {}
        on_ms = int(cfg.get('alarm_on_delay_ms', DEFAULT_ON_DELAY_MS))
        off_ms = int(cfg.get('alarm_off_delay_ms', DEFAULT_OFF_DELAY_MS))
        return on_ms / 1000.0, off_ms / 1000.0

    def set_motion(self, camera, is_motion, source='unknown', immediate=False):
        """Record a raw motion observation and emit events on debounced transitions.

        camera: VirtualONVIFCamera instance (used for id, name, motion config).
        is_motion: bool — current raw observation.
        source: short label ('synthetic', 'detector') for logging only.
        immediate: if True, bypass debounce and commit immediately. Used by the
            synthetic motion endpoints (whose purpose is to validate NVR event
            ingestion) and for clean shutdown.
        """
        cid = camera.id
        if immediate:
            with self._lock:
                entry = self._states.setdefault(
                    cid, {"state": False, "pending": None, "pending_since": 0.0}
                )
                if entry["state"] == is_motion and entry["pending"] is None:
                    return entry["state"]
                entry["state"] = is_motion
                entry["pending"] = None
                entry["pending_since"] = 0.0
            self._emit(camera, is_motion, source)
            return is_motion

        on_delay, off_delay = self._get_delays(camera)
        now = time.time()

        with self._lock:
            entry = self._states.setdefault(
                cid, {"state": False, "pending": None, "pending_since": 0.0}
            )
            current_state = entry["state"]

            # Nothing to do if observation matches steady state and no pending change.
            if is_motion == current_state and entry["pending"] is None:
                return current_state

            # If a pending change exists in the opposite direction, cancel it.
            if entry["pending"] is not None and entry["pending"] != is_motion:
                entry["pending"] = None
                entry["pending_since"] = 0.0
                if is_motion == current_state:
                    return current_state

            # Start a pending transition if none is in flight.
            if entry["pending"] is None and is_motion != current_state:
                entry["pending"] = is_motion
                entry["pending_since"] = now

            # Commit if enough time has elapsed.
            required = on_delay if is_motion else off_delay
            if entry["pending"] is not None and (now - entry["pending_since"]) >= required:
                entry["state"] = entry["pending"]
                entry["pending"] = None
                entry["pending_since"] = 0.0
                committed = entry["state"]
                emit = True
            else:
                committed = entry["state"]
                emit = False

        if emit:
            self._emit(camera, committed, source)
        return committed

    def get_state(self, camera_id):
        with self._lock:
            entry = self._states.get(camera_id)
            return bool(entry and entry["state"])

    def _emit(self, camera, is_motion, source):
        bus = get_event_bus()
        bus.publish(camera.id, TOPIC_CELL_MOTION, {"IsMotion": is_motion})
        bus.publish(camera.id, TOPIC_MOTION_ALARM, {"State": is_motion})
        if getattr(camera, 'debug_mode', False):
            verb = "STARTED" if is_motion else "STOPPED"
            print(f"  [Motion] {camera.name}: motion {verb} (source={source})")


_CONTROLLER = MotionController()


def get_motion_controller():
    return _CONTROLLER
