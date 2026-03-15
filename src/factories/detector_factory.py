class DriftDetectorFactory:
    @staticmethod
    def create(config):
        name = config["drift_detector"]["name"]

        # if name == "adwin":
        #     return ADWINDetector(delta=config["drift_detector"]["delta"])
        # if name == "ddm":
        #     return DDMDetector()
        # if name == "page_hinkley":
        #     return PageHinkleyDetector(
        #         delta=config["drift_detector"]["delta"],
        #         threshold=config["drift_detector"]["threshold"]
        #     )
        raise ValueError(name)