# messages/message_types/incrementing_id_message.py
from typing import ClassVar
from mcpcli.messages.message_types.json_rpc_message import JSONRPCMessage

class IncrementingIDMessage(JSONRPCMessage):
    counter: ClassVar[int] = 0

    @classmethod
    def load_counter(cls, value: int):
        cls.counter = value

    @classmethod
    def save_counter(cls) -> int:
        return cls.counter

    def __init__(self, prefix: str, method: str, start_id: int = None, **kwargs):
        if start_id is not None:
            type(self).counter = start_id
        else:
            type(self).counter += 1

        message_id = f"{prefix}-{type(self).counter}"
        super().__init__(method=method, id=message_id, **kwargs)
