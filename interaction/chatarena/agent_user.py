from typing import List, Union
from tenacity import RetryError
import logging
import uuid
from abc import abstractmethod
import asyncio

from .backends import IntelligenceBackend, load_backend
from .message import Message, SYSTEM_NAME
from .config import AgentConfig, Configurable, BackendConfig

# A special signal sent by the player to indicate that it is not possible to continue the conversation, and it requests to end the conversation.
# It contains a random UUID string to avoid being exploited by any of the players.
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class Agent(Configurable):

    @abstractmethod
    def __init__(self, name: str, role_desc: str, global_prompt: str = None, request_msg: Message = None, *args, **kwargs):
        super().__init__(name=name, role_desc=role_desc, global_prompt=global_prompt, request_msg=request_msg, **kwargs)
        self.name = name
        self.role_desc = role_desc
        self.global_prompt = global_prompt
        self.request_msg = request_msg


class Player(Agent):
    """
    Player of the game. It can takes the observation from the environment and return an action
    """

    def __init__(self, name: str, role_desc: str, backend: Union[BackendConfig, IntelligenceBackend],
                 global_prompt: str = None, request_msg: Message = None, **kwargs):

        if isinstance(backend, BackendConfig):
            backend_config = backend
            backend = load_backend(backend_config)
        elif isinstance(backend, IntelligenceBackend):
            backend_config = backend.to_config()
        else:
            raise ValueError(f"backend must be a BackendConfig or an IntelligenceBackend, but got {type(backend)}")

        assert name != SYSTEM_NAME, f"Player name cannot be {SYSTEM_NAME}, which is reserved for the system."

        # Register the fields in the _config
        super().__init__(name=name, role_desc=role_desc, backend=backend_config,
                         global_prompt=global_prompt, request_msg=request_msg, **kwargs)

        self.backend = backend

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            backend=self.backend.to_config(),
            global_prompt=self.global_prompt,
        )

    def act(self, observation: List[Message]) -> str:
        """
        Call the agents to generate a response (equivalent to taking an action).
        """
        try:
            response = self.backend.query(agent_name=self.name, role_desc=self.role_desc,
                                          history_messages=observation, global_prompt=self.global_prompt,
                                          request_msg=self.request_msg)
        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e.last_attempt.exception()}. "
                            f"Sending signal to end the conversation.")
            response = SIGNAL_END_OF_CONVERSATION

        return response

    def __call__(self, observation: List[Message]) -> str:
        return self.act(observation)

    async def async_act(self, observation: List[Message]) -> str:
        """
        Async call the agents to generate a response (equivalent to taking an action).
        """
        try:
            response = self.backend.async_query(agent_name=self.name, role_desc=self.role_desc,
                                                history_messages=observation, global_prompt=self.global_prompt,
                                                request_msg=self.request_msg)
        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e.last_attempt.exception()}. "
                            f"Sending signal to end the conversation.")
            response = SIGNAL_END_OF_CONVERSATION

        return response

    def reset(self):
        self.backend.reset()
