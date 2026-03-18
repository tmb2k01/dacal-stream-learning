from engine.stream_engine import DriftAdaptiveActiveLearningEngine
from factories import (
    CalibratorFactory,
    DatasetFactory,
    DriftDetectorFactory,
    ModelFactory,
    PolicyFactory,
    TaskFactory,
)

# config = load_config("configs/experiment.yaml")
config = None

task = TaskFactory.create(config)
dataset = DatasetFactory.create(config)
predictor = ModelFactory.create(config)
calibrator = CalibratorFactory.create(config, task=task)
active_policy = PolicyFactory.create(config)
drift_detector = DriftDetectorFactory.create(config)
label_oracle = dataset  # Assuming the dataset can serve as a label oracle for simplicity
engine = DriftAdaptiveActiveLearningEngine(
    task=task,
    predictor=predictor,
    calibrator=calibrator,
    active_policy=active_policy,
    drift_detector=drift_detector,
    label_oracle=label_oracle,
)
