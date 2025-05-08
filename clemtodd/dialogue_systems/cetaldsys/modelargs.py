from typing import Dict, Optional
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to utilize.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the HuggingFace model."}
    )
    model_name_or_path_agent: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the HuggingFace model for the agent"}
    )
    use_int8: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use int8 model or not."}
    )
    use_deepspeed: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use deepspeed model or not."}
    )