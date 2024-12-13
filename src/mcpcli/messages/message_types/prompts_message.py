# messages/message_types/prompts_message.py
from mcpcli.messages.message_types.incrementing_id_message import IncrementingIDMessage

class PromptsListMessage(IncrementingIDMessage):
    def __init__(self, start_id: int = None, **kwargs):
        super().__init__(prefix="prompts-list", method="prompts/list", start_id=start_id, **kwargs)
