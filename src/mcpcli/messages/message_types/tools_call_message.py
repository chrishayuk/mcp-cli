# messages/message_types/tools_call_message.py
from mcpcli.messages.message_types.incrementing_id_message import IncrementingIDMessage

class CallToolMessage(IncrementingIDMessage):
    def __init__(self, tool_name: str, arguments: dict, start_id: int = None, **kwargs):
        super().__init__(prefix="tools-call", method="tools/call", start_id=start_id, params={"name": tool_name, "arguments": arguments}, **kwargs)
