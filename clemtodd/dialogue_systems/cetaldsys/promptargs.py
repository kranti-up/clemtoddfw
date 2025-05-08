from typing import Dict, Optional
from transformers import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class PromptingArguments(TrainingArguments):
    """
    Arguments pertraining to the prompting pipeline.
    """
    output_dir: Optional[str] = field(
        default="games/todsystem/dialogue_systems/cetaldsys/",
        metadata={"help": "Output directory"},
    )
    task: Optional[str] = field(
        default="dst",
        metadata={"help": "Task to perform"}
    )
    max_requests_per_minute: Optional[int] = field(
        default=20,
        metadata={"help": "Max number of requests for OpenAI API."}
    )
    openai_api_key_name: Optional[str] = field(
        default="OPENAI_API_KEY",
        metadata={"help": "OpenAI API key name."}
    )