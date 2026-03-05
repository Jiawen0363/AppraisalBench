from typing import List, Union

from .backends import IntelligenceBackend, load_backend
from .config import AgentConfig, BackendConfig
from .agent import Player


class Assistant(Player):
    """
    Vanilla LLM agent for the other party in the dialogue.
    Same behaviour as Player; name defaults to "Assistant".
    """

    def __init__(
        self,
        role_desc: str,
        backend: Union[BackendConfig, IntelligenceBackend],
        global_prompt: str = None,
        name: str = "Assistant",
        **kwargs,
    ):
        super().__init__(
            name=name,
            role_desc=role_desc,
            backend=backend,
            global_prompt=global_prompt,
            **kwargs,
        )

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            backend=self.backend.to_config(),
            global_prompt=self.global_prompt,
        )
