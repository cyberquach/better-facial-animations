"""Conservative, independently tested shortcuts and surface review gates."""


def can_skip_speech(before, after):
    return (before.get("fingerprint") is not None
            and before["fingerprint"] == after.get("fingerprint")
            and before.get("has_repair_state") is False
            and after.get("has_repair_state") is False)


def surface_regressions(before, after):
    if not before.get("edges") or before.get("edges") != after.get("edges"):
        return ["upper_surface_unmeasured"]
    reasons = []
    if after["p95_strain"] > before["p95_strain"] + 0.001:
        reasons.append("upper_surface_p95_worse")
    if after["over_double_length"] > before["over_double_length"]:
        reasons.append("upper_surface_severe_count_worse")
    return reasons
