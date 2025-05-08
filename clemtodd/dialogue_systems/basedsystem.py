from abc import ABC, abstractmethod
from typing import Dict



class DialogueSystem(ABC):
    """Abstract base class for all dialogue systems."""

    def __init__(self, **kwargs):
        """Allow subclasses to accept parameters during initialization."""
        pass

    @abstractmethod
    def process_user_input(self, user_input: str, current_turn: int) -> str:
        """Processes user input and returns system response."""
        pass

    @abstractmethod
    def get_booking_data(self) -> Dict:
        """Returns generated slots."""
        pass    