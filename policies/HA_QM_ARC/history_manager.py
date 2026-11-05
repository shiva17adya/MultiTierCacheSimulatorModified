"""Workload History Manager for HA-QM-ARC.

Maintains bounded O(1) streaming metrics over sliding windows:
- Rolling hit/miss ratio & O(1) hit gradient
- Frequency evolution & burst concentration
- Configurable sequential access / scan streak detection
- QoS priority penalty stress
- Hit location tracking (T1, T2, B1, B2, MISS)
"""

from collections import deque, Counter
import re
from typing import Optional, Dict


class WorkloadHistoryManager:
    """Bounded, streaming history manager operating in strictly O(1) time per request."""

    def __init__(
        self,
        window_size: int = 100,
        burst_window_size: int = 20,
        scan_streak_threshold: int = 5,
        enable_scan_detection: bool = False,
    ):
        """Initialize the history manager.

        Args:
            window_size: Sliding window capacity W for general metrics (default 100).
            burst_window_size: Sub-window capacity for rapid repeat/burst detection (default 20).
            scan_streak_threshold: Number of consecutive sequential accesses to trigger scan detection (default 5).
            enable_scan_detection: Flag enabling sequential scan tracking. Should only be True for
                workloads with verified ordered identifiers (e.g. chunked video streams). Default False.
        """
        self.window_size = max(10, int(window_size))
        self.half_window_size = max(5, self.window_size // 2)
        self.burst_window_size = max(5, min(int(burst_window_size), self.window_size))
        self.scan_streak_threshold = max(2, int(scan_streak_threshold))
        self.enable_scan_detection = bool(enable_scan_detection)

        # Circular ring buffers (O(1) append & pop)
        self.hit_window = deque(maxlen=self.window_size)
        self.hp_req_window = deque(maxlen=self.window_size)
        self.hp_miss_window = deque(maxlen=self.window_size)
        self.burst_window = deque(maxlen=self.burst_window_size)
        self.location_window = deque(maxlen=self.window_size)

        # Two half-window deques for strictly O(1) gradient computation
        self.recent_half_window = deque(maxlen=self.half_window_size)
        self.prior_half_window = deque(maxlen=self.half_window_size)

        # O(1) tracking scalar counters
        self.total_hits_in_window = 0
        self.hp_req_in_window = 0
        self.hp_miss_in_window = 0
        self.burst_counts = Counter()
        self.location_counts = Counter()

        self.recent_half_hits = 0
        self.prior_half_hits = 0

        # Sequential scan tracking
        self.last_item_id: Optional[int] = None
        self.current_scan_streak = 0
        self.max_observed_scan_streak = 0

    def reset(self):
        """Reset all state counters and buffers for clean experiment isolation."""
        self.hit_window.clear()
        self.hp_req_window.clear()
        self.hp_miss_window.clear()
        self.burst_window.clear()
        self.location_window.clear()

        self.recent_half_window.clear()
        self.prior_half_window.clear()

        self.total_hits_in_window = 0
        self.hp_req_in_window = 0
        self.hp_miss_in_window = 0
        self.burst_counts.clear()
        self.location_counts.clear()

        self.recent_half_hits = 0
        self.prior_half_hits = 0

        self.last_item_id = None
        self.current_scan_streak = 0
        self.max_observed_scan_streak = 0

    def _extract_numeric_id(self, packet_name: str) -> Optional[int]:
        """Extract trailing or standalone numeric ID only from structured identifiers."""
        if not self.enable_scan_detection:
            return None

        # Pure integer name
        try:
            return int(packet_name)
        except (ValueError, TypeError):
            pass

        # Structured naming with explicit sequential pattern: e.g. 'video_chunk_123', 'block_42'
        name_str = str(packet_name)
        match = re.search(r"(?:chunk|block|item|seq|page|seg)[_\-/]?(\d+)$", name_str, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None

        return None

    def record_access(
        self,
        packet_name: str,
        priority: str,
        is_hit: bool,
        hit_location: str = "MISS",
    ):
        """Record an incoming request and its completed cache outcome in O(1) time.

        Args:
            packet_name: Name/identifier of the accessed packet.
            priority: Priority string ('h' or 'high', 'l' or 'low').
            is_hit: True if hit in T1 or T2; False on cache miss or ghost hit.
            hit_location: Completed outcome location ('T1', 'T2', 'B1', 'B2', 'MISS').
        """
        is_hp = str(priority).lower().startswith("h")
        hit_val = 1 if is_hit else 0
        hp_req_val = 1 if is_hp else 0
        hp_miss_val = 1 if (is_hp and not is_hit) else 0

        # Maintain hit window (O(1))
        if len(self.hit_window) == self.hit_window.maxlen:
            old_hit = self.hit_window[0]
            self.total_hits_in_window -= old_hit
        self.hit_window.append(hit_val)
        self.total_hits_in_window += hit_val

        # Maintain two half-windows for strictly O(1) gradient computation
        if len(self.recent_half_window) == self.recent_half_window.maxlen:
            transferred = self.recent_half_window[0]
            self.recent_half_hits -= transferred
            if len(self.prior_half_window) == self.prior_half_window.maxlen:
                oldest = self.prior_half_window[0]
                self.prior_half_hits -= oldest
            self.prior_half_window.append(transferred)
            self.prior_half_hits += transferred
        self.recent_half_window.append(hit_val)
        self.recent_half_hits += hit_val

        # Maintain HP request and miss windows (O(1))
        if len(self.hp_req_window) == self.hp_req_window.maxlen:
            old_hp_req = self.hp_req_window[0]
            self.hp_req_in_window -= old_hp_req
        self.hp_req_window.append(hp_req_val)
        self.hp_req_in_window += hp_req_val

        if len(self.hp_miss_window) == self.hp_miss_window.maxlen:
            old_hp_miss = self.hp_miss_window[0]
            self.hp_miss_in_window -= old_hp_miss
        self.hp_miss_window.append(hp_miss_val)
        self.hp_miss_in_window += hp_miss_val

        # Maintain hit location window (O(1))
        loc = str(hit_location).upper()
        if len(self.location_window) == self.location_window.maxlen:
            old_loc = self.location_window[0]
            self.location_counts[old_loc] -= 1
        self.location_window.append(loc)
        self.location_counts[loc] += 1

        # Maintain burst sub-window (O(1))
        if len(self.burst_window) == self.burst_window.maxlen:
            old_burst_key = self.burst_window[0]
            self.burst_counts[old_burst_key] -= 1
            if self.burst_counts[old_burst_key] <= 0:
                del self.burst_counts[old_burst_key]
        self.burst_window.append(packet_name)
        self.burst_counts[packet_name] += 1

        # Maintain sequential scan streak only if enabled and strictly consecutive
        if self.enable_scan_detection:
            curr_id = self._extract_numeric_id(packet_name)
            if curr_id is not None and self.last_item_id is not None:
                # Strictly consecutive forward progression (repeats are bursts, not scans)
                if curr_id == self.last_item_id + 1:
                    self.current_scan_streak += 1
                else:
                    self.current_scan_streak = 0
            else:
                self.current_scan_streak = 0

            self.last_item_id = curr_id
            if self.current_scan_streak > self.max_observed_scan_streak:
                self.max_observed_scan_streak = self.current_scan_streak
        else:
            self.current_scan_streak = 0
            self.last_item_id = None

    def get_rolling_ghr(self) -> float:
        """Return the current rolling global hit ratio in [0.0, 1.0] (strictly O(1))."""
        n = len(self.hit_window)
        if n == 0:
            return 0.0
        return self.total_hits_in_window / float(n)

    def get_ghr_gradient(self) -> float:
        """Return hit-rate gradient between recent and prior half-windows in strictly O(1) time."""
        n_recent = len(self.recent_half_window)
        n_prior = len(self.prior_half_window)
        if n_prior == 0 or n_recent == 0:
            return 0.0
        ghr_recent = self.recent_half_hits / float(n_recent)
        ghr_prior = self.prior_half_hits / float(n_prior)
        return ghr_recent - ghr_prior

    def get_burst_score(self) -> float:
        """Compute the concentration of repeated accesses in the burst window in [0.0, 1.0] (strictly O(1))."""
        n = len(self.burst_window)
        if n <= 1:
            return 0.0
        num_unique = len(self.burst_counts)
        duplicates = n - num_unique
        return min(1.0, max(0.0, duplicates / float(n - 1)))

    def get_scan_score(self) -> float:
        """Compute sequential scan streak intensity in [0.0, 1.0] (strictly O(1))."""
        if not self.enable_scan_detection:
            return 0.0
        return min(1.0, self.current_scan_streak / float(self.scan_streak_threshold))

    def get_hp_stress(self) -> float:
        """Compute high-priority penalty stress in [0.0, 1.0] (strictly O(1))."""
        if self.hp_req_in_window == 0:
            return 0.0
        return min(1.0, max(0.0, self.hp_miss_in_window / float(self.hp_req_in_window)))

    def get_hit_locations_breakdown(self) -> Dict[str, float]:
        """Return normalized distribution of access outcomes over the window in O(1) time."""
        n = len(self.location_window)
        if n == 0:
            return {"T1": 0.0, "T2": 0.0, "B1": 0.0, "B2": 0.0, "MISS": 0.0}
        return {
            loc: self.location_counts.get(loc, 0) / float(n)
            for loc in ["T1", "T2", "B1", "B2", "MISS"]
        }
