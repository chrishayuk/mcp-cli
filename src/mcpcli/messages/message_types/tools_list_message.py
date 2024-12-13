# messages/message_types/tools_List_message.py
from mcpcli.messages.message_types.incrementing_id_message import IncrementingIDMessage

class ToolsListMessage(IncrementingIDMessage):
    def __init__(self, start_id: int = None, **kwargs):
        super().__init__(prefix="tools-list", method="tools/list", start_id=start_id, **kwargs)