"""userating -- turn free-text user feedback into 1-5 star ratings."""

from .metrics import eval_metrics, eval_report
from .model import SentimentClassifier, load, save

__version__ = "1.0.0"

__all__ = ["SentimentClassifier", "save", "load", "eval_metrics", "eval_report", "__version__"]
