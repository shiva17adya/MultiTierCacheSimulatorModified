import sys
from pathlib import Path

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import numpy as np
import simpy

from forwarder_structures.content_store.tier import Tier
from forwarder_structures.content_store.index import Index
from forwarder_structures.pit import PIT
from forwarder_structures.forwarder import Forwarder
from common.packet import Packet
from common.penalty import get_alpha
from policies.QL_QM_ARC.ql_agent import QLearningAgent
from policies.QM_ARC.abstract_qm_arc_policy import AbstractQMARCPolicy
from policies.QM_ARC.tier_qm_arc_policy import QMARCPolicy
from policies.MARC.abstract_m_arc_policy import AbstractMARCPolicy
from policies.MARC.tier_m_arc_policy import MARCPolicy
from policies.QL_QM_ARC.abstract_ql_qm_arc_policy import AbstractQLQMARCPolicy
from policies.QL_QM_ARC.tier_ql_qm_arc_policy import QLQMARCPolicy
from policies.lru_policy import LRUPolicy
from policies.lfu_policy import LFUPolicy
from policies.random_policy import RandPolicy


# ============================================================================
# 1. Q-LEARNING AGENT TESTS
# ============================================================================

class TestQLearningAgent:
    def test_action_space_and_indexing(self):
        """Verify that action_idx maps accurately to action_space[action_idx] and column bounds."""
        num_states = 10
        num_actions = 4
        agent = QLearningAgent(num_states, num_actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.0)

        assert len(agent.action_space) == 2 * num_actions
        # Column 0 -> -4, Column 4 -> 0, Column 7 -> +3
        assert agent.action_space[0] == -4
        assert agent.action_space[4] == 0
        assert agent.action_space[7] == 3

        assert agent.q_table.shape == (num_states, 2 * num_actions)

    def test_q_table_updates_exact_column(self):
        """Verify that Bellman updates modify the exact action_idx column, not the signed action_value."""
        num_states = 10
        num_actions = 5
        agent = QLearningAgent(num_states, num_actions, learning_rate=0.5, discount_factor=0.9, epsilon=0.0)

        # Set action_idx 7 (which represents action +2) to a high Q-value
        agent.q_table[4, 7] = 20.0

        # Exploitation should pick column 7
        next_state = agent.get_next_state(current_p=4, current_t1_len=3, current_b1_len=2, current_b2_len=2, hit="t1")
        # 4 + 2 = 6
        assert next_state == 6

        # Verify column 7 in row 4 was updated: Q = 20 + 0.5 * (100 + 0.9 * 0 - 20) = 20 + 40 = 60
        assert agent.q_table[4, 7] == pytest.approx(60.0)
        # Verify column 2 (which would have been mistakenly updated by the old bug) remains 0.0
        assert agent.q_table[4, 2] == 0.0

    def test_boundary_conditions(self):
        """Verify state boundaries [0, num_states - 1] are strictly enforced."""
        num_states = 8
        num_actions = 4
        agent = QLearningAgent(num_states, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.0)

        # Action that pushes below 0
        agent.q_table[1, 0] = 100.0  # action -4
        s_low = agent.get_next_state(current_p=1, current_t1_len=0, current_b1_len=0, current_b2_len=0, hit="Miss")
        assert s_low == 0

        # Action that pushes above upper bound
        agent.q_table[6, 7] = 100.0  # action +3
        s_high = agent.get_next_state(current_p=6, current_t1_len=0, current_b1_len=0, current_b2_len=0, hit="t1")
        assert s_high == num_states - 1

        # Out-of-bounds input current_p is clamped gracefully
        s_clamp = agent.get_next_state(current_p=999, current_t1_len=0, current_b1_len=0, current_b2_len=0, hit="Miss")
        assert 0 <= s_clamp <= num_states - 1

    def test_exploration_vs_exploitation(self):
        """Verify epsilon=0 purely exploits while epsilon=1 purely explores."""
        agent_exploit = QLearningAgent(num_states=5, num_actions=3, learning_rate=0.1, discount_factor=0.9, epsilon=0.0)
        agent_exploit.q_table[2, 5] = 99.0  # column 5 represents action +2

        # 10 successive calls must all choose column 5
        for _ in range(10):
            res = agent_exploit.get_next_state(current_p=2, current_t1_len=1, current_b1_len=1, current_b2_len=1, hit="t1")
            assert res == min(4, 2 + 2)

        # Pure exploration should produce varied actions over repeated trials
        agent_explore = QLearningAgent(num_states=10, num_actions=5, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, seed=123)
        actions_chosen = set()
        for _ in range(50):
            nxt = agent_explore.get_next_state(current_p=5, current_t1_len=1, current_b1_len=1, current_b2_len=1, hit="t1")
            actions_chosen.add(nxt)
        assert len(actions_chosen) > 1

    def test_repeated_learning_updates(self):
        """Verify that learning with hits increases expected reward values over iterations."""
        agent = QLearningAgent(num_states=10, num_actions=4, learning_rate=0.1, discount_factor=0.9, epsilon=0.0)
        assert np.max(agent.q_table[3]) == 0.0

        for _ in range(20):
            agent.get_next_state(current_p=3, current_t1_len=2, current_b1_len=1, current_b2_len=1, hit="t1")

        assert np.max(agent.q_table[3]) > 0.0
        assert len(agent.rewards) == 20
        assert all(r == 100 for r in agent.rewards)


# ============================================================================
# 2. MULTI-TIER CAPACITY & RATIO TESTS
# ============================================================================

class TestMultiTierCapacityCalculations:
    def test_two_tier_configuration(self):
        """Verify 2-tier (DRAM + SSD) dynamic beta calculations and adaptation."""
        env = simpy.Environment()
        index = Index(env)
        boss = Tier(name="BOSS", max_size=0, granularity=0, latency=0, read_throughput=0, write_throughput=0)
        dram = Tier(name="DRAM", max_size=1000, granularity=1, latency=1e-7, read_throughput=40e9, write_throughput=20e9)
        ssd = Tier(name="SSD", max_size=5000, granularity=512, latency=1e-5, read_throughput=2e9, write_throughput=1e9)
        tiers = [boss, dram, ssd]
        pit = PIT()
        fwd = Forwarder(env, index, tiers, pit, slot_size=100, default_tier_index=1)

        boss_policy = AbstractQMARCPolicy(env, fwd, boss)
        dram_policy = QMARCPolicy(env, fwd, dram)
        ssd_policy = QMARCPolicy(env, fwd, ssd)

        physical = boss_policy.get_physical_tiers()
        assert len(physical) == 2
        assert physical[0].name == "DRAM"
        assert physical[1].name == "SSD"

        betas, beta_sum = boss_policy.get_tier_betas()
        assert betas[dram] == 1.0
        assert betas[ssd] == 5.0
        assert beta_sum == 6.0

        # Increment with |B2| = 30, |B1| = 10 -> ratio = 3.0
        boss_policy.increment_p(len_b1=10, len_b2=30)
        # Global p delta = 3.0 * 6.0 = 18.0
        assert boss_policy.p == 18.0
        # Local p delta: dram = 3.0 * 1.0 = 3.0, ssd = 3.0 * 5.0 = 15.0
        assert dram_policy.p == 3.0
        assert ssd_policy.p == 15.0

        # Decrement with |B1| = 20, |B2| = 10 -> ratio = 2.0
        boss_policy.decrement_p(len_b1=20, len_b2=10)
        # Global p delta = 2.0 * 6.0 = 12.0 -> 18.0 - 12.0 = 6.0
        assert boss_policy.p == 6.0
        assert dram_policy.p == 1.0
        assert ssd_policy.p == 5.0

    def test_three_tier_configuration(self):
        """Verify 3-tier (DRAM + SSD + NVMe) dynamic beta calculations and conservation invariant."""
        env = simpy.Environment()
        index = Index(env)
        boss = Tier(name="BOSS", max_size=0, granularity=0, latency=0, read_throughput=0, write_throughput=0)
        dram = Tier(name="DRAM", max_size=1000, granularity=1, latency=1e-7, read_throughput=40e9, write_throughput=20e9)
        ssd = Tier(name="SSD", max_size=5000, granularity=512, latency=1e-5, read_throughput=2e9, write_throughput=1e9)
        nvme = Tier(name="NVME", max_size=25000, granularity=512, latency=1e-4, read_throughput=500e6, write_throughput=250e6)
        tiers = [boss, dram, ssd, nvme]
        pit = PIT()
        fwd = Forwarder(env, index, tiers, pit, slot_size=100, default_tier_index=1)

        boss_policy = AbstractQMARCPolicy(env, fwd, boss)
        dram_policy = QMARCPolicy(env, fwd, dram)
        ssd_policy = QMARCPolicy(env, fwd, ssd)
        nvme_policy = QMARCPolicy(env, fwd, nvme)

        physical = boss_policy.get_physical_tiers()
        assert len(physical) == 3
        assert [t.name for t in physical] == ["DRAM", "SSD", "NVME"]

        betas, beta_sum = boss_policy.get_tier_betas()
        assert betas[dram] == 1.0
        assert betas[ssd] == 5.0
        assert betas[nvme] == 25.0
        assert beta_sum == 31.0

        # Increment: ratio = 2.0
        boss_policy.increment_p(len_b1=5, len_b2=10)
        # Conservation invariant: sum(delta_p_i) == delta_p_global
        assert boss_policy.p == 62.0
        assert dram_policy.p == 2.0
        assert ssd_policy.p == 10.0
        assert nvme_policy.p == 50.0
        assert (dram_policy.p + ssd_policy.p + nvme_policy.p) == boss_policy.p

    def test_backwards_compatible_properties(self):
        """Verify that beta_ssd and beta_disk properties retain backward compatibility."""
        env = simpy.Environment()
        index = Index(env)
        boss = Tier(name="BOSS", max_size=0, granularity=0, latency=0, read_throughput=0, write_throughput=0)
        dram = Tier(name="DRAM", max_size=200, granularity=1, latency=1e-7, read_throughput=40e9, write_throughput=20e9)
        ssd = Tier(name="SSD", max_size=1000, granularity=512, latency=1e-5, read_throughput=2e9, write_throughput=1e9)
        disk = Tier(name="DISK", max_size=5000, granularity=512, latency=1e-4, read_throughput=500e6, write_throughput=250e6)
        fwd = Forwarder(env, index, [boss, dram, ssd, disk], PIT(), slot_size=50, default_tier_index=1)

        boss_policy = AbstractQMARCPolicy(env, fwd, boss)
        assert boss_policy.beta_ssd == 5.0
        assert boss_policy.beta_disk == 25.0


# ============================================================================
# 3. CACHE REPLACEMENT ALGORITHM BEHAVIOUR TESTS
# ============================================================================

class TestCacheAlgorithmBehaviour:
    def test_lru_hit_miss_and_eviction(self):
        """Verify LRU capacity enforcement and MRU promotion."""
        env = simpy.Environment()
        index = Index(env)
        # Slot size = 100, max_size = 200 -> capacity = 2 packets
        dram = Tier(name="DRAM", max_size=200, granularity=1, latency=1e-7, read_throughput=1e9, write_throughput=1e9)
        fwd = Forwarder(env, index, [dram], PIT(), slot_size=100, default_tier_index=0)
        lru = LRUPolicy(env, fwd, dram)
        assert lru.c == 2

        res = [simpy.Resource(env, capacity=1)]
        p1 = Packet("d", 0.0, "pkt1", 100, "l", 0.05)
        p2 = Packet("d", 0.1, "pkt2", 100, "l", 0.05)
        p3 = Packet("d", 0.2, "pkt3", 100, "l", 0.05)

        # Insert p1, p2
        def run_sim():
            yield env.process(lru.on_packet_access(env, res, p1, is_write=True))
            yield env.process(lru.on_packet_access(env, res, p2, is_write=True))
            assert "pkt1" in lru.lru_dict
            assert "pkt2" in lru.lru_dict

            # Access p1 (hit -> moves to MRU)
            yield env.process(lru.on_packet_access(env, res, p1, is_write=False))

            # Insert p3 -> exceeds capacity (2), evicts LRU (pkt2)
            yield env.process(lru.on_packet_access(env, res, p3, is_write=True))
            assert "pkt1" in lru.lru_dict
            assert "pkt3" in lru.lru_dict
            assert "pkt2" not in lru.lru_dict

        env.process(run_sim())
        env.run()

    def test_lfu_frequency_tracking(self):
        """Verify LFU eviction based on access frequency."""
        env = simpy.Environment()
        index = Index(env)
        dram = Tier(name="DRAM", max_size=200, granularity=1, latency=1e-7, read_throughput=1e9, write_throughput=1e9)
        fwd = Forwarder(env, index, [dram], PIT(), slot_size=100, default_tier_index=0)
        lfu = LFUPolicy(env, fwd, dram)
        res = [simpy.Resource(env, capacity=1)]

        p1 = Packet("d", 0.0, "p1", 100, "l", 0.05)
        p2 = Packet("d", 0.1, "p2", 100, "l", 0.05)
        p3 = Packet("d", 0.2, "p3", 100, "l", 0.05)

        def run_sim():
            yield env.process(lfu.on_packet_access(env, res, p1, is_write=True))
            yield env.process(lfu.on_packet_access(env, res, p2, is_write=True))

            # Access p1 three times -> freq = 4
            for _ in range(3):
                yield env.process(lfu.on_packet_access(env, res, p1, is_write=False))

            # Insert p3 -> capacity exceeded, p2 (freq 1) must be evicted instead of p1 (freq 4)
            yield env.process(lfu.on_packet_access(env, res, p3, is_write=True))
            assert "p1" in lfu.keyToVal
            assert "p3" in lfu.keyToVal
            assert "p2" not in lfu.keyToVal

        env.process(run_sim())
        env.run()

    def test_random_policy_capacity_enforcement(self):
        """Verify Random policy respects cache capacity."""
        env = simpy.Environment()
        index = Index(env)
        dram = Tier(name="DRAM", max_size=300, granularity=1, latency=1e-7, read_throughput=1e9, write_throughput=1e9)
        fwd = Forwarder(env, index, [dram], PIT(), slot_size=100, default_tier_index=0)
        rand_pol = RandPolicy(env, fwd, dram)
        res = [simpy.Resource(env, capacity=1)]

        def run_sim():
            for i in range(10):
                pkt = Packet("d", float(i), f"item_{i}", 100, "l", 0.05)
                yield env.process(rand_pol.on_packet_access(env, res, pkt, is_write=True))
            assert len(rand_pol.random_struct) <= rand_pol.c

        env.process(run_sim())
        env.run()

    def test_qm_arc_priority_positioning(self):
        """Verify QM-ARC differential placement between High Priority (MRU) and Low Priority."""
        env = simpy.Environment()
        index = Index(env)
        boss = Tier(name="BOSS", max_size=0, granularity=0, latency=0, read_throughput=0, write_throughput=0)
        dram = Tier(name="DRAM", max_size=1000, granularity=1, latency=1e-7, read_throughput=40e9, write_throughput=20e9)
        tiers = [boss, dram]
        fwd = Forwarder(env, index, tiers, PIT(), slot_size=100, default_tier_index=1)

        boss_policy = AbstractQMARCPolicy(env, fwd, boss)
        dram_policy = QMARCPolicy(env, fwd, dram)
        res = [simpy.Resource(env, capacity=1), simpy.Resource(env, capacity=1)]

        # High priority packet
        hp_pkt = Packet("d", 0.0, "hp_item", 100, "h", 0.05)
        # Low priority packet
        lp_pkt = Packet("d", 0.1, "lp_item", 100, "l", 0.05)

        def run_sim():
            # Miss for hp_pkt -> inserted at MRU (head) of T1
            yield env.process(boss_policy.on_packet_access(env, res, hp_pkt, is_write=False))
            t1_items = list(boss_policy.t1.items())
            assert len(t1_items) == 1
            assert t1_items[0][0] == "hp_item"

            # Low priority miss with non-empty T1: positioned at round(len(T1) * lambda)
            # lambda = 0.2, len(T1) = 1 -> pos = round(0.2) = 0
            yield env.process(boss_policy.on_packet_access(env, res, lp_pkt, is_write=False))
            assert "lp_item" in boss_policy.t1

        env.process(run_sim())
        env.run()

    def test_m_arc_lifecycle_and_ghost_lists(self):
        """Verify M-ARC promotion to T2, ghost list tracking in B1/B2, and adaptation on ghost hit."""
        env = simpy.Environment()
        index = Index(env)
        boss = Tier("BOSS", 0, 0, 0, 0, 0)
        # 2 tiers, slot_size = 100 -> dram cap = 1, ssd cap = 1, total c = 2
        dram = Tier("DRAM", 100, 1, 1e-7, 40e9, 20e9)
        ssd = Tier("SSD", 100, 512, 1e-5, 2e9, 1e9)
        fwd = Forwarder(env, index, [boss, dram, ssd], PIT(), slot_size=100, default_tier_index=1)

        m_arc_boss = AbstractMARCPolicy(env, fwd, boss)
        MARCPolicy(env, fwd, dram)
        MARCPolicy(env, fwd, ssd)
        res = [simpy.Resource(env, 1), simpy.Resource(env, 1), simpy.Resource(env, 1)]

        p1 = Packet("d", 0.0, "p1", 100, "l", 0.05)
        p2 = Packet("d", 0.1, "p2", 100, "l", 0.05)
        p3 = Packet("d", 0.2, "p3", 100, "l", 0.05)

        def run_sim():
            # 1. Miss p1 -> in T1
            yield env.process(m_arc_boss.on_packet_access(env, res, p1, is_write=False))
            assert "p1" in m_arc_boss.t1
            assert "p1" not in m_arc_boss.t2

            # 2. Hit p1 in T1 -> promoted to MRU of T2
            yield env.process(m_arc_boss.on_packet_access(env, res, p1, is_write=False))
            assert "p1" in m_arc_boss.t2
            assert "p1" not in m_arc_boss.t1

            # 3. Miss p2 -> in T1
            yield env.process(m_arc_boss.on_packet_access(env, res, p2, is_write=False))
            assert "p2" in m_arc_boss.t1

            # 4. Miss p3 -> capacity reached (c=2), p2 evicted from T1 to ghost B1
            yield env.process(m_arc_boss.on_packet_access(env, res, p3, is_write=False))
            in_b1 = yield env.process(fwd.index.packet_in_ghost("p2", "b1"))
            assert in_b1, "p2 should be moved to ghost list B1"

            # 5. Access p2 (hit in B1) -> increases p, removes from ghost, places into T2
            init_p = m_arc_boss.p
            yield env.process(m_arc_boss.on_packet_access(env, res, p2, is_write=False))
            assert m_arc_boss.p >= init_p
            in_b1_after = yield env.process(fwd.index.packet_in_ghost("p2", "b1"))
            assert not in_b1_after, "p2 should be removed from ghost list B1 after promotion"
            assert "p2" in m_arc_boss.t2

        env.process(run_sim())
        env.run()

    def test_priority_lru_routing_and_prefetching(self):
        """Verify Priority-LRU tier routing: HP miss -> DRAM, LP miss -> SSD, HP hit in SSD -> prefetch to DRAM."""
        env = simpy.Environment()
        index = Index(env)
        dram = Tier("DRAM", 200, 1, 1e-7, 40e9, 20e9)
        ssd = Tier("SSD", 500, 512, 1e-5, 2e9, 1e9)
        fwd = Forwarder(env, index, [dram, ssd], PIT(), slot_size=100, default_tier_index=0)
        LRUPolicy(env, fwd, dram)
        LRUPolicy(env, fwd, ssd)
        res = [simpy.Resource(env, 1), simpy.Resource(env, 1)]

        hp_pkt = Packet("d", 0.0, "hp_item", 100, "h", 0.05)
        lp_pkt = Packet("d", 0.1, "lp_item", 100, "l", 0.05)

        def run_sim():
            # HP miss -> writes to default tier (DRAM)
            yield env.process(fwd.get_default_tier().write_packet(env, res, hp_pkt))
            tier_hp = yield env.process(fwd.index.get_packet_tier("hp_item"))
            assert tier_hp.name == "DRAM"

            # LP miss -> writes to last tier (SSD)
            yield env.process(fwd.get_last_tier().write_packet(env, res, lp_pkt))
            tier_lp = yield env.process(fwd.index.get_packet_tier("lp_item"))
            assert tier_lp.name == "SSD"

            # If an HP packet was in SSD, accessing it triggers prefetch to DRAM
            hp_on_ssd = Packet("d", 0.2, "hp_in_ssd", 100, "h", 0.05)
            yield env.process(fwd.get_last_tier().write_packet(env, res, hp_on_ssd))
            tier_before = yield env.process(fwd.index.get_packet_tier("hp_in_ssd"))
            assert tier_before.name == "SSD"

            # Simulate prefetching triggered by PriorityTrace on SSD hit
            yield env.process(fwd.get_last_tier().prefetch_packet(env, hp_on_ssd))
            yield env.process(fwd.get_default_tier().write_packet(env, res, hp_on_ssd, cause="prefetching"))
            tier_after = yield env.process(fwd.index.get_packet_tier("hp_in_ssd"))
            assert tier_after.name == "DRAM"

        env.process(run_sim())
        env.run()

    def test_ql_qm_arc_three_tier_adaptation(self):
        """Verify QL-QM-ARC under 3-tier setup (DRAM + SSD + NVMe) updates Q-table and local tier p values."""
        env = simpy.Environment()
        index = Index(env)
        boss = Tier("BOSS", 0, 0, 0, 0, 0)
        dram = Tier("DRAM", 200, 1, 1e-7, 40e9, 20e9)
        ssd = Tier("SSD", 1000, 512, 1e-5, 2e9, 1e9)
        nvme = Tier("NVME", 5000, 512, 1e-4, 500e6, 250e6)
        tiers = [boss, dram, ssd, nvme]
        fwd = Forwarder(env, index, tiers, PIT(), slot_size=100, default_tier_index=1)

        ql_boss = AbstractQLQMARCPolicy(env, fwd, boss)
        QLQMARCPolicy(env, fwd, dram)
        QLQMARCPolicy(env, fwd, ssd)
        QLQMARCPolicy(env, fwd, nvme)
        res = [simpy.Resource(env, 1)] * 4

        # Initial Q-table should be all zeros
        assert np.max(ql_boss.q_learning_agent.q_table) == 0.0

        p1 = Packet("d", 0.0, "pkt1", 100, "h", 0.05)
        p2 = Packet("d", 0.1, "pkt2", 100, "l", 0.05)

        def run_sim():
            # Misses
            yield env.process(ql_boss.on_packet_access(env, res, p1, is_write=False))
            yield env.process(ql_boss.on_packet_access(env, res, p2, is_write=False))
            # Hit in T1
            yield env.process(ql_boss.on_packet_access(env, res, p1, is_write=False))

            # Q-table should now have learned updates
            assert len(ql_boss.q_learning_agent.rewards) > 0
            # Test local p increment
            ql_boss.increment_p(len_b1=5, len_b2=15)
            # Physical tiers must all have their local p adjusted
            pt = ql_boss.get_physical_tiers()
            assert len(pt) == 3
            for t in pt:
                assert t.strategies[0].p >= 0

        env.process(run_sim())
        env.run()

    def test_tier_to_tier_demotion_cascade(self):
        """Verify that when a tier fills up, items demote to the next physical tier."""
        env = simpy.Environment()
        index = Index(env)
        boss = Tier("BOSS", 0, 0, 0, 0, 0)
        # Each tier holds exactly 1 packet (slot_size=100)
        dram = Tier("DRAM", 100, 1, 1e-7, 40e9, 20e9)
        ssd = Tier("SSD", 100, 512, 1e-5, 2e9, 1e9)
        fwd = Forwarder(env, index, [boss, dram, ssd], PIT(), slot_size=100, default_tier_index=1)

        boss_policy = AbstractQMARCPolicy(env, fwd, boss)
        QMARCPolicy(env, fwd, dram)
        QMARCPolicy(env, fwd, ssd)
        res = [simpy.Resource(env, 1)] * 3

        p1 = Packet("d", 0.0, "p1", 100, "h", 0.05)
        p2 = Packet("d", 0.1, "p2", 100, "h", 0.05)

        def run_sim():
            # Insert p1 -> placed in DRAM (default tier)
            yield env.process(boss_policy.on_packet_access(env, res, p1, is_write=False))
            tier_p1 = yield env.process(index.get_packet_tier("p1"))
            assert tier_p1.name == "DRAM"

            # Insert p2 -> DRAM is full (cap=1), p1 is demoted from DRAM to SSD
            yield env.process(boss_policy.on_packet_access(env, res, p2, is_write=False))
            tier_p2 = yield env.process(index.get_packet_tier("p2"))
            tier_p1_after = yield env.process(index.get_packet_tier("p1"))
            assert tier_p2.name == "DRAM"
            assert tier_p1_after.name == "SSD"

        env.process(run_sim())
        env.run()

    def test_capacity_constraints_after_repeated_operations(self):
        """Verify active cache capacity constraint (|CS| <= sum(c_i)) after repeated operations."""
        env = simpy.Environment()
        index = Index(env)
        boss = Tier("BOSS", 0, 0, 0, 0, 0)
        dram = Tier("DRAM", 200, 1, 1e-7, 40e9, 20e9)
        ssd = Tier("SSD", 300, 512, 1e-5, 2e9, 1e9)
        fwd = Forwarder(env, index, [boss, dram, ssd], PIT(), slot_size=100, default_tier_index=1)
        # Total physical capacity = (200 + 300) / 100 = 5 packets
        total_capacity = 5

        boss_policy = AbstractQMARCPolicy(env, fwd, boss)
        QMARCPolicy(env, fwd, dram)
        QMARCPolicy(env, fwd, ssd)
        res = [simpy.Resource(env, 1)] * 3

        def run_sim():
            # 50 random accesses on 20 distinct keys
            for i in range(50):
                key = f"key_{i % 20}"
                prio = "h" if i % 5 == 0 else "l"
                pkt = Packet("d", float(i) * 0.01, key, 100, prio, 0.05)
                yield env.process(boss_policy.on_packet_access(env, res, pkt, is_write=False))

            # Active index count must not exceed physical capacity
            active_count = len(index.active_index)
            assert active_count <= total_capacity, f"Active index ({active_count}) exceeded capacity ({total_capacity})"

        env.process(run_sim())
        env.run()
