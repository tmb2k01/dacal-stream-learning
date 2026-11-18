from models.base import BasePredictor
from models.river_predictor import RiverClassifierPredictor
from models.simple_cnn import SimpleCNN
from models.sklearn_predictor import SklearnClassifierPredictor

__all__ = [
    "BasePredictor",
    "RiverClassifierPredictor",
    "SimpleCNN",
    "SklearnClassifierPredictor",
]
