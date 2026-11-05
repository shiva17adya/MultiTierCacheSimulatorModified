"""Adaptive Controller for HA-QM-ARC.

Computes proactive parameter modulations for:
1. Target partition boundary p (balancing recency T1 and frequency T2)
2. Dynamic QoS insertion parameter lambda_effective (based on high-priority SLA stress)
"""

from typing import Dict, Any


class AdaptiveController:
    """Controller that converts workload history metrics into cache control decisions."""

    def __init__(
        self,
        omega_burst: float = 0.15,
        omega_scan: float = 0.20,
        omega_grad: float = 0.10,
        kappa_qos: float = 0.50,
        scan_damp_threshold: float = 0.80,
    ):
        """Initialize the adaptive controller.

        Args:
            omega_burst: Sensitivity weight for burst concentration (shifts capacity to T2).
            omega_scan: Sensitivity weight for sequential scans (throttles T1 expansion).
            omega_grad: Sensitivity weight for rolling hit gradient (rewards positive trend).
            kappa_qos: Attenuation factor for QoS high-priority stress on low-priority placement.
            scan_damp_threshold: Threshold above which scan protection is activated.
        """
        self.omega_burst = float(omega_burst)
        self.omega_scan = float(omega_scan)
        self.omega_grad = float(omega_grad)
        self.kappa_qos = float(kappa_qos)
        self.scan_damp_threshold = float(scan_damp_threshold)

    def compute_delta_p_history(
        self,
        c: int,
        burst_score: float,
        scan_score: float,
        ghr_gradient: float,
    ) -> float:
        """Compute the momentum bias delta_p_history from streaming workload metrics.

        Formula:
            delta_p = round(c * [-omega_burst * burst - omega_scan * scan + omega_grad * grad])

        Clamped to [-0.3 * c, 0.3 * c] to prevent abrupt partition oscillation.
        """
        if c <= 0:
            return 0.0

        raw_bias = (
            -self.omega_burst * burst_score
            - self.omega_scan * scan_score
            + self.omega_grad * ghr_gradient
        )

        delta = round(float(c) * raw_bias)
        max_shift = max(1.0, 0.30 * float(c))
        return max(-max_shift, min(max_shift, float(delta)))

    def compute_effective_lambda(
        self,
        base_lambda: float,
        hp_stress: float,
    ) -> float:
        """Compute dynamic QoS placement parameter lambda_effective.

        Formula:
            lambda_effective = base_lambda * (1.0 - kappa_qos * hp_stress)

        Clamped to [base_lambda * 0.20, base_lambda] to ensure a valid non-zero insertion index.
        """
        base = max(0.01, float(base_lambda))
        stress = max(0.0, min(1.0, float(hp_stress)))

        effective = base * (1.0 - self.kappa_qos * stress)
        min_lambda = base * 0.20
        return max(min_lambda, min(base, effective))

    def get_effective_p(
        self,
        base_p: float,
        c: int,
        burst_score: float,
        scan_score: float,
        ghr_gradient: float,
    ) -> float:
        """Calculate effective target p bounded strictly in [0, c]."""
        delta = self.compute_delta_p_history(c, burst_score, scan_score, ghr_gradient)
        effective_p = base_p + delta
        return max(0.0, min(float(c), effective_p))

    def is_scan_active(self, scan_score: float) -> bool:
        """Determine whether sequential scan streak protection should be triggered."""
        return scan_score >= self.scan_damp_threshold

    def get_diagnostics(
        self,
        c: int,
        base_p: float,
        base_lambda: float,
        burst_score: float,
        scan_score: float,
        ghr_gradient: float,
        hp_stress: float,
    ) -> Dict[str, Any]:
        """Return diagnostic metrics dictionary for debugging and analysis."""
        delta_p = self.compute_delta_p_history(c, burst_score, scan_score, ghr_gradient)
        effective_p = self.get_effective_p(base_p, c, burst_score, scan_score, ghr_gradient)
        effective_lambda = self.compute_effective_lambda(base_lambda, hp_stress)
        return {
            "burst_score": burst_score,
            "scan_score": scan_score,
            "ghr_gradient": ghr_gradient,
            "hp_stress": hp_stress,
            "delta_p_history": delta_p,
            "base_p": base_p,
            "effective_p": effective_p,
            "base_lambda": base_lambda,
            "effective_lambda": effective_lambda,
            "scan_active": self.is_scan_active(scan_score),
        }
