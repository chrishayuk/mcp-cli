# messages/message_types/tools_messages.py
from mcpcli.messages.message_types.incrementing_id_message import IncrementingIDMessage
from mcpcli.messages.message_types.json_rpc_message import JSONRPCMessage

class ToolsListMessage(IncrementingIDMessage):
    def __init__(self, start_id: int = None, **kwargs):
        super().__init__(prefix="tools-list", method="tools/list", start_id=start_id, **kwargs)
        
class CallToolMessage(IncrementingIDMessage):
    def __init__(self, tool_name: str, arguments: dict, start_id: int = None, **kwargs):
        super().__init__(prefix="tools-call", method="tools/call", start_id=start_id, params={"name": tool_name, "arguments": arguments}, **kwargs)

class ToolsListChangedMessage(JSONRPCMessage):
    def __init__(self, **kwargs):
        # A notification has no 'id' field.
        super().__init__(method="notifications/tools/list_changed", id=None, **kwargs)
