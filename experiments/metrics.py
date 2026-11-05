from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class TierMetrics:
    """Metrics recorded for an individual physical storage tier."""
    tier_name: str
    max_size: int
    used_size: int
    occupancy_ratio: float
    hits: int
    hits_hp: int
    hits_lp: int
    # Contribution to global hit ratio: hits / total_requests
    hit_contribution: float
    hit_contribution_hp: float
    hit_contribution_lp: float
    # Conditional hit ratio: hits / requests reaching this tier
    requests_reaching_tier: int
    conditional_hit_ratio: float
    # Backward compatibility alias
    @property
    def hit_ratio(self) -> float:
        return self.hit_contribution

    number_of_reads: int = 0
    number_of_writes: int = 0
    time_spent_reading: float = 0.0
    time_spent_writing: float = 0.0
    number_of_evictions_from: int = 0
    number_of_evictions_to: int = 0


@dataclass
class SimulationMetrics:
    """Standardized performance metrics aggregating both direct and derived measurements."""
    # Direct / Measured Counts
    total_requests: int
    high_priority_requests: int
    low_priority_requests: int
    total_hits: int
    total_misses: int
    total_hp_hits: int
    total_lp_hits: int

    # Penalties (Directly accumulated from SLA penalty function)
    penalty_hp: float
    penalty_lp: float
    total_penalty: float

    # Derived Hit Ratios (Safe with 0 denominators)
    global_hit_ratio: float
    high_priority_hit_ratio: float
    low_priority_hit_ratio: float

    # Retrieval Latencies (Seconds)
    avg_retrieval_latency: float
    avg_retrieval_latency_hp: float
    avg_retrieval_latency_lp: float

    # Execution overhead
    simulation_wall_time_seconds: float

    # Per-tier breakdowns
    tier_breakdown: List[TierMetrics] = field(default_factory=list)

    @classmethod
    def from_simulation(cls, forwarder, nb_interests: int, nb_hp: int, nb_lp: int, wall_time: float) -> "SimulationMetrics":
        """Extract and compute standardized metrics from Forwarder and Tier objects."""
        physical_tiers = [t for t in forwarder.tiers if t.name != "BOSS" and t.max_size > 0]
        if not physical_tiers:
            physical_tiers = forwarder.tiers

        total_hits = sum(t.chr for t in physical_tiers)
        total_hp_hits = sum(t.chr_hpc for t in physical_tiers)
        total_lp_hits = sum(t.chr_lpc for t in physical_tiers)
        total_misses = sum(t.cmr for t in physical_tiers)

        # Penalties are recorded in default tier by simulator convention
        default_tier = forwarder.get_default_tier()
        pen_hp = float(default_tier.penalty_hpc)
        pen_lp = float(default_tier.penalty_lpc)

        total_hp_time = sum(float(t.high_p_data_retrieval_time) for t in physical_tiers)
        total_lp_time = sum(float(t.low_p_data_retrieval_time) for t in physical_tiers)

        # Safe derived divisions
        ghr = float(total_hits) / float(nb_interests) if nb_interests > 0 else 0.0
        hphr = float(total_hp_hits) / float(nb_hp) if nb_hp > 0 else 0.0
        lphr = float(total_lp_hits) / float(nb_lp) if nb_lp > 0 else 0.0

        avg_lat_hp = total_hp_time / float(nb_hp) if nb_hp > 0 else 0.0
        avg_lat_lp = total_lp_time / float(nb_lp) if nb_lp > 0 else 0.0
        avg_lat = (total_hp_time + total_lp_time) / float(nb_interests) if nb_interests > 0 else 0.0

        tier_metrics_list = []
        cumulative_hits_before = 0
        for t in physical_tiers:
            cap = float(t.max_size * t.target_occupation) if t.max_size > 0 else 1.0
            occ = float(t.used_size) / cap if cap > 0 else 0.0

            # Contribution to global hit ratio: tier hits / total requests
            t_contrib = float(t.chr) / float(nb_interests) if nb_interests > 0 else 0.0
            t_contrib_hp = float(t.chr_hpc) / float(nb_hp) if nb_hp > 0 else 0.0
            t_contrib_lp = float(t.chr_lpc) / float(nb_lp) if nb_lp > 0 else 0.0

            # Requests reaching this tier: total requests minus hits in higher-speed tiers
            reqs_reaching = max(0, nb_interests - cumulative_hits_before)
            cond_hr = float(t.chr) / float(reqs_reaching) if reqs_reaching > 0 else 0.0
            cumulative_hits_before += t.chr

            tier_metrics_list.append(
                TierMetrics(
                    tier_name=t.name,
                    max_size=t.max_size,
                    used_size=t.used_size,
                    occupancy_ratio=occ,
                    hits=t.chr,
                    hits_hp=t.chr_hpc,
                    hits_lp=t.chr_lpc,
                    hit_contribution=t_contrib,
                    hit_contribution_hp=t_contrib_hp,
                    hit_contribution_lp=t_contrib_lp,
                    requests_reaching_tier=reqs_reaching,
                    conditional_hit_ratio=cond_hr,
                    number_of_reads=t.number_of_reads,
                    number_of_writes=t.number_of_write,
                    time_spent_reading=float(t.time_spent_reading),
                    time_spent_writing=float(t.time_spent_writing),
                    number_of_evictions_from=t.number_of_eviction_from_this_tier,
                    number_of_evictions_to=t.number_of_eviction_to_this_tier,
                )
            )

        return cls(
            total_requests=nb_interests,
            high_priority_requests=nb_hp,
            low_priority_requests=nb_lp,
            total_hits=total_hits,
            total_misses=total_misses,
            total_hp_hits=total_hp_hits,
            total_lp_hits=total_lp_hits,
            penalty_hp=pen_hp,
            penalty_lp=pen_lp,
            total_penalty=pen_hp + pen_lp,
            global_hit_ratio=ghr,
            high_priority_hit_ratio=hphr,
            low_priority_hit_ratio=lphr,
            avg_retrieval_latency=avg_lat,
            avg_retrieval_latency_hp=avg_lat_hp,
            avg_retrieval_latency_lp=avg_lat_lp,
            simulation_wall_time_seconds=wall_time,
            tier_breakdown=tier_metrics_list,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "high_priority_requests": self.high_priority_requests,
            "low_priority_requests": self.low_priority_requests,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "total_hp_hits": self.total_hp_hits,
            "total_lp_hits": self.total_lp_hits,
            "penalty_hp": self.penalty_hp,
            "penalty_lp": self.penalty_lp,
            "total_penalty": self.total_penalty,
            "global_hit_ratio": self.global_hit_ratio,
            "high_priority_hit_ratio": self.high_priority_hit_ratio,
            "low_priority_hit_ratio": self.low_priority_hit_ratio,
            "avg_retrieval_latency": self.avg_retrieval_latency,
            "avg_retrieval_latency_hp": self.avg_retrieval_latency_hp,
            "avg_retrieval_latency_lp": self.avg_retrieval_latency_lp,
            "simulation_wall_time_seconds": self.simulation_wall_time_seconds,
            "tier_breakdown": [
                {
                    "tier_name": t.tier_name,
                    "max_size": t.max_size,
                    "used_size": t.used_size,
                    "occupancy_ratio": t.occupancy_ratio,
                    "hits": t.hits,
                    "hits_hp": t.hits_hp,
                    "hits_lp": t.hits_lp,
                    "hit_contribution": t.hit_contribution,
                    "hit_contribution_hp": t.hit_contribution_hp,
                    "hit_contribution_lp": t.hit_contribution_lp,
                    "requests_reaching_tier": t.requests_reaching_tier,
                    "conditional_hit_ratio": t.conditional_hit_ratio,
                    "hit_ratio": t.hit_ratio,  # backward compatibility alias
                    "number_of_reads": t.number_of_reads,
                    "number_of_writes": t.number_of_writes,
                    "time_spent_reading": t.time_spent_reading,
                    "time_spent_writing": t.time_spent_writing,
                    "evictions_from": t.number_of_evictions_from,
                    "evictions_to": t.number_of_evictions_to,
                }
                for t in self.tier_breakdown
            ],
        }
