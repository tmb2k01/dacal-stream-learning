from active_learning import UncertaintySamplingPolicy, HybridActivePolicy

class PolicyFactory:
    @staticmethod
    def create(config):
        name = config["active_policy"]["name"]

        if name == "uncertainty":
            return UncertaintySamplingPolicy(
                threshold=config["active_policy"]["threshold"]
            )
        if name == "hybrid":
            return HybridActivePolicy(
                unc_threshold=config["active_policy"]["unc_threshold"],
                set_size_threshold=config["active_policy"]["set_size_threshold"]
            )

        raise ValueError(name)