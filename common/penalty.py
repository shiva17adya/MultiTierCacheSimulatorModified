min_high_prio_penalty = 0
mid_high_prio_penalty = 50
max_high_prio_penalty = 75

min_low_prio_penalty = 0
mid_low_prio_penalty = 10
max_low_prio_penalty = 15

low_prio_disk_penalty = 0
high_prio_disk_penalty = 5


def set_penalty_lambda(lambda_val: float):
    """Set the QoS penalty lambda ratio (ratio of LP penalty to HP penalty)."""
    global mid_low_prio_penalty, max_low_prio_penalty
    mid_low_prio_penalty = mid_high_prio_penalty * float(lambda_val)
    max_low_prio_penalty = max_high_prio_penalty * float(lambda_val)


def get_alpha():
    alpha = max_low_prio_penalty / max_high_prio_penalty if max_high_prio_penalty > 0 else 0.2
    return alpha


def get_penalty(response_time, priority) -> float:
    if priority == "h":
        if response_time == 0.0:
            return float(high_prio_disk_penalty)
        if response_time < 0.02:
            return float(min_high_prio_penalty)
        if response_time < 0.15:
            return float(mid_high_prio_penalty)
        elif response_time >= 0.15:
            return float(max_high_prio_penalty)

    elif priority == "l":
        if response_time == 0.0:
            return float(low_prio_disk_penalty)
        if response_time < 0.02:
            return float(min_low_prio_penalty)
        if response_time < 0.15:
            return float(mid_low_prio_penalty)
        elif response_time >= 0.15:
            return float(max_low_prio_penalty)

    return 0.0
