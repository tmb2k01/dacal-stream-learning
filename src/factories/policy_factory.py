from active_learning import (
    AlwaysQueryPolicy,
    HybridActivePolicy,
    RandomQueryPolicy,
    UncertaintySamplingPolicy,
)


class PolicyFactory:
    @staticmethod
    def create(config):
        policy_cfg = config.get("active_policy") or config.get("active_learning") or {}
        if not policy_cfg or not policy_cfg.get("enabled", True):
            return None

        name = policy_cfg.get("name")
        if name in {None, "none"}:
            return None
        if name == "uncertainty":
            return UncertaintySamplingPolicy(threshold=policy_cfg["threshold"])
        if name == "hybrid":
            return HybridActivePolicy(
                unc_threshold=policy_cfg["unc_threshold"],
                set_size_threshold=policy_cfg["set_size_threshold"],
            )
        if name == "random":
            return RandomQueryPolicy(
                rate=policy_cfg["rate"],
                burst_rate=policy_cfg.get("burst_rate"),
                seed=policy_cfg.get("seed", config.get("seed", 0)),
            )
        if name in {"always", "full"}:
            return AlwaysQueryPolicy()

        raise ValueError(f"Unsupported active policy: {name!r}")
