class ModelFactory:
    @staticmethod
    def create(config):
        task_type = config["task"]["type"]
        model_name = config["model"]["name"]

        # if task_type == "classification" and model_name == "mlp":
        #     backbone = MLPClassifier(...)
        # elif task_type == "regression" and model_name == "mlp":
        #     backbone = MLPRegressor(...)
        # elif task_type == "timeseries" and model_name == "tcn":
        #     backbone = TCNForecaster(...)
        # elif task_type == "timeseries" and model_name == "transformer":
        #     backbone = TemporalTransformer(...)
        # else:
        #     raise ValueError("Unsupported model/task pair")

        # return LightningOnlineModel(
        #     backbone=backbone,
        #     task_type=task_type,
        #     lr=config["model"]["lr"]
        # )