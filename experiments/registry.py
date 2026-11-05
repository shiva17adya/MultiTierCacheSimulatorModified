from dataclasses import dataclass
from typing import Dict, Type, Optional, List

from traces.trace_reading.arc_trace import ARCTrace
from traces.trace_reading.common_trace import CommonTrace
from traces.trace_reading.priority_trace import PriorityTrace
from traces.trace_reading.trace import Trace

from policies.QM_ARC.abstract_qm_arc_policy import AbstractQMARCPolicy
from policies.QM_ARC.tier_qm_arc_policy import QMARCPolicy
from policies.QL_QM_ARC.abstract_ql_qm_arc_policy import AbstractQLQMARCPolicy
from policies.QL_QM_ARC.tier_ql_qm_arc_policy import QLQMARCPolicy
from policies.HA_QM_ARC.abstract_ha_qm_arc_policy import AbstractHAQMARCPolicy
from policies.HA_QM_ARC.tier_ha_qm_arc_policy import TierHAQMARCPolicy
from policies.MARC.abstract_m_arc_policy import AbstractMARCPolicy
from policies.MARC.tier_m_arc_policy import MARCPolicy
from policies.lru_policy import LRUPolicy
from policies.lfu_policy import LFUPolicy
from policies.random_policy import RandPolicy
from policies.policy import Policy


@dataclass(frozen=True)
class AlgorithmDescriptor:
    """Metadata and class bindings required to instantiate and execute a caching algorithm."""
    name: str
    category: str  # 'arc' (needs BOSS coordinator), 'standard', or 'priority_routing'
    coordinator_class: Optional[Type[Policy]]
    tier_policy_class: Type[Policy]
    trace_reader_class: Type[Trace]
    description: str


class AlgorithmRegistry:
    """Central registry decoupling algorithm implementations from simulation orchestration."""

    _REGISTRY: Dict[str, AlgorithmDescriptor] = {}
    _ALIASES: Dict[str, str] = {}

    @classmethod
    def register(cls, canonical_name: str, descriptor: AlgorithmDescriptor, aliases: Optional[List[str]] = None):
        """Register an algorithm descriptor under its canonical name and optional aliases."""
        cls._REGISTRY[canonical_name] = descriptor
        cls._ALIASES[canonical_name.lower().replace("-", "").replace("_", "")] = canonical_name
        if aliases:
            for alias in aliases:
                norm = alias.lower().replace("-", "").replace("_", "")
                cls._ALIASES[norm] = canonical_name

    @classmethod
    def get(cls, name: str) -> AlgorithmDescriptor:
        """Retrieve an algorithm descriptor by name (case-insensitive, alias-friendly)."""
        norm = name.strip().lower().replace("-", "").replace("_", "")
        if norm in cls._ALIASES:
            canonical = cls._ALIASES[norm]
            return cls._REGISTRY[canonical]

        available = list(cls._REGISTRY.keys())
        raise ValueError(
            f"Unsupported algorithm '{name}'. Available algorithms: {available}"
        )

    @classmethod
    def list_algorithms(cls) -> List[str]:
        """Return the canonical names of all registered algorithms."""
        return sorted(list(cls._REGISTRY.keys()))

    @classmethod
    def is_registered(cls, name: str) -> bool:
        norm = name.strip().lower().replace("-", "").replace("_", "")
        return norm in cls._ALIASES


# ============================================================================
# Core Algorithm Registrations
# ============================================================================

AlgorithmRegistry.register(
    "QM-ARC",
    AlgorithmDescriptor(
        name="QM-ARC",
        category="arc",
        coordinator_class=AbstractQMARCPolicy,
        tier_policy_class=QMARCPolicy,
        trace_reader_class=ARCTrace,
        description="QoS-aware Multi-tier Adaptive Replacement Cache with priority-biased positioning",
    ),
    aliases=["QMARC", "QM_ARC"],
)

AlgorithmRegistry.register(
    "QL-QM-ARC",
    AlgorithmDescriptor(
        name="QL-QM-ARC",
        category="arc",
        coordinator_class=AbstractQLQMARCPolicy,
        tier_policy_class=QLQMARCPolicy,
        trace_reader_class=ARCTrace,
        description="Reinforcement Learning (Q-learning) enhanced QoS-aware Multi-tier ARC",
    ),
    aliases=["QLQMARC", "QL_QM_ARC"],
)

AlgorithmRegistry.register(
    "HA-QM-ARC",
    AlgorithmDescriptor(
        name="HA-QM-ARC",
        category="arc",
        coordinator_class=AbstractHAQMARCPolicy,
        tier_policy_class=TierHAQMARCPolicy,
        trace_reader_class=ARCTrace,
        description="History-Aware Adaptive QM-ARC with proactive p adaptation and dynamic QoS placement",
    ),
    aliases=["HAQMARC", "HA_QM_ARC"],
)

AlgorithmRegistry.register(
    "M-ARC",
    AlgorithmDescriptor(
        name="M-ARC",
        category="arc",
        coordinator_class=AbstractMARCPolicy,
        tier_policy_class=MARCPolicy,
        trace_reader_class=ARCTrace,
        description="Standard Multi-tier Adaptive Replacement Cache without QoS management",
    ),
    aliases=["MARC", "M_ARC"],
)

AlgorithmRegistry.register(
    "Priority-LRU",
    AlgorithmDescriptor(
        name="Priority-LRU",
        category="priority_routing",
        coordinator_class=None,
        tier_policy_class=LRUPolicy,
        trace_reader_class=PriorityTrace,
        description="Priority-partitioned LRU routing HP misses to DRAM and LP misses to SSD",
    ),
    aliases=["PriorityLRU", "Priority_LRU"],
)

AlgorithmRegistry.register(
    "LRU",
    AlgorithmDescriptor(
        name="LRU",
        category="standard",
        coordinator_class=None,
        tier_policy_class=LRUPolicy,
        trace_reader_class=CommonTrace,
        description="Classic Least Recently Used cache replacement",
    ),
    aliases=["LeastRecentlyUsed"],
)

AlgorithmRegistry.register(
    "LFU",
    AlgorithmDescriptor(
        name="LFU",
        category="standard",
        coordinator_class=None,
        tier_policy_class=LFUPolicy,
        trace_reader_class=CommonTrace,
        description="Classic Least Frequently Used cache replacement",
    ),
    aliases=["LeastFrequentlyUsed"],
)

AlgorithmRegistry.register(
    "Random",
    AlgorithmDescriptor(
        name="Random",
        category="standard",
        coordinator_class=None,
        tier_policy_class=RandPolicy,
        trace_reader_class=CommonTrace,
        description="Random eviction replacement baseline",
    ),
    aliases=["Rand"],
)
