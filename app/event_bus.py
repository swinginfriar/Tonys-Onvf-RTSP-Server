"""ONVIF event bus for per-camera motion and notification events.

Holds a small in-memory queue of recent events per camera and tracks PullPoint
subscriptions. ONVIF clients (e.g. UniFi Protect) subscribe via the events
service, then poll PullMessages to drain events. Internal producers (synthetic
motion endpoints, motion detection workers) call publish() to add events.
"""

import threading
import time
import uuid
from collections import deque

# Per-camera ring buffer size. Cameras generate at most a few events per second
# during motion, so 200 is generous and bounds memory.
MAX_EVENTS_PER_CAMERA = 200

# Default subscription lifetime if the client does not specify one
DEFAULT_TIMEOUT_SECONDS = 60

# Subscription expiry sweep interval
SWEEP_INTERVAL_SECONDS = 30


class Event:
    """A single ONVIF notification event."""

    def __init__(self, camera_id, topic, data, utc_time=None):
        self.id = None  # Assigned by EventBus when published
        self.camera_id = camera_id
        self.topic = topic
        self.data = dict(data)  # e.g. {"IsMotion": True}
        self.utc_time = utc_time or time.time()


class Subscription:
    """Tracks a PullPoint subscription for one camera."""

    def __init__(self, camera_id, timeout_seconds):
        self.id = uuid.uuid4().hex
        self.camera_id = camera_id
        self.created_at = time.time()
        self.expires_at = self.created_at + timeout_seconds
        # Cursor is exclusive: events with id > cursor have not been delivered yet.
        # New subscriptions start at the current head so they only see future events.
        self.cursor = 0
        # Lock + condition so PullMessages can block until events arrive
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def renew(self, timeout_seconds):
        self.expires_at = time.time() + timeout_seconds

    def is_expired(self):
        return time.time() >= self.expires_at


class EventBus:
    """Process-wide event bus shared by all cameras."""

    def __init__(self):
        self._lock = threading.RLock()
        # camera_id -> deque[Event] (newest at the right)
        self._events = {}
        # camera_id -> monotonically increasing event id counter
        self._next_event_id = {}
        # subscription_id -> Subscription
        self._subscriptions = {}
        # Wakeup conditions per camera so all subscribers can be notified at once
        self._camera_conds = {}
        self._sweeper_started = False

    def _ensure_camera(self, camera_id):
        if camera_id not in self._events:
            self._events[camera_id] = deque(maxlen=MAX_EVENTS_PER_CAMERA)
            self._next_event_id[camera_id] = 0
            self._camera_conds[camera_id] = threading.Condition(self._lock)

    def publish(self, camera_id, topic, data, utc_time=None):
        """Add an event for this camera and wake any blocked PullMessages calls.

        utc_time is optional — caller can pass a specific UTC epoch timestamp
        to backdate (or future-date) the event, which controls where the
        event lands on the NVR's timeline. Defaults to "now" when omitted.
        """
        with self._lock:
            self._ensure_camera(camera_id)
            self._next_event_id[camera_id] += 1
            event = Event(camera_id, topic, data, utc_time=utc_time)
            event.id = self._next_event_id[camera_id]
            self._events[camera_id].append(event)
            self._camera_conds[camera_id].notify_all()

    def create_subscription(self, camera_id, timeout_seconds=None):
        timeout_seconds = timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        with self._lock:
            self._ensure_camera(camera_id)
            sub = Subscription(camera_id, timeout_seconds)
            # Start at the current head so the subscriber only sees future events
            sub.cursor = self._next_event_id[camera_id]
            self._subscriptions[sub.id] = sub
            self._maybe_start_sweeper()
            return sub

    def get_subscription(self, subscription_id):
        with self._lock:
            return self._subscriptions.get(subscription_id)

    def renew(self, subscription_id, timeout_seconds=None):
        timeout_seconds = timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        with self._lock:
            sub = self._subscriptions.get(subscription_id)
            if not sub:
                return None
            sub.renew(timeout_seconds)
            return sub

    def unsubscribe(self, subscription_id):
        with self._lock:
            sub = self._subscriptions.pop(subscription_id, None)
            if sub:
                # Wake any waiting pull so it returns immediately
                self._camera_conds[sub.camera_id].notify_all()
            return sub is not None

    def pull_messages(self, subscription_id, wait_seconds, max_messages):
        """Block up to wait_seconds for new events, then return a batch.

        Returns (subscription, events_list). subscription is None if not found.
        """
        with self._lock:
            sub = self._subscriptions.get(subscription_id)
            if not sub:
                return None, []
            if sub.is_expired():
                self._subscriptions.pop(subscription_id, None)
                return None, []

            cond = self._camera_conds[sub.camera_id]
            deadline = time.time() + max(0, wait_seconds)
            while True:
                pending = [e for e in self._events[sub.camera_id] if e.id > sub.cursor]
                if pending:
                    batch = pending[:max_messages]
                    sub.cursor = batch[-1].id
                    return sub, batch
                remaining = deadline - time.time()
                if remaining <= 0:
                    return sub, []
                cond.wait(timeout=remaining)

    def set_synchronization_point(self, subscription_id):
        """Reset the cursor to the most recent event so the next pull starts fresh."""
        with self._lock:
            sub = self._subscriptions.get(subscription_id)
            if not sub:
                return False
            sub.cursor = self._next_event_id.get(sub.camera_id, 0)
            return True

    def _maybe_start_sweeper(self):
        if self._sweeper_started:
            return
        self._sweeper_started = True
        t = threading.Thread(target=self._sweep_loop, daemon=True)
        t.start()

    def _sweep_loop(self):
        while True:
            time.sleep(SWEEP_INTERVAL_SECONDS)
            try:
                self._expire_stale()
            except Exception as e:
                print(f"  [EventBus] Sweeper error: {e}")

    def _expire_stale(self):
        with self._lock:
            stale = [sid for sid, s in self._subscriptions.items() if s.is_expired()]
            for sid in stale:
                sub = self._subscriptions.pop(sid, None)
                if sub:
                    self._camera_conds[sub.camera_id].notify_all()


# Process-wide singleton. The ONVIFService instances all share this bus.
_BUS = EventBus()


def get_event_bus():
    return _BUS
