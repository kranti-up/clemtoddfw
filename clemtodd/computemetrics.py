from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#from clemgame import get_logger


import logging

logger = logging.getLogger(__name__)


class ComputeMetrics:
    def __init__(self):
        pass

    def _setto_lower(self, slots: dict) -> dict:
        return {
            str(key).lower(): str(value).lower()
            for key, value in slots.items()
        }

    def run(self, results):
        slots_gt = results["slots_gt"]
        slots_gen = results["slots_gen"]


        



