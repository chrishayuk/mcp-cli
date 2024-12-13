# messages/message_types/resources_messages.py
from typing import ClassVar
from mcpcli.messages.message_types.incrementing_id_message import IncrementingIDMessage

class ResourcesListMessage(IncrementingIDMessage):
    def __init__(self, start_id: int = None, **kwargs):
        super().__init__(prefix="resources-list", method="resources/list", start_id=start_id, **kwargs)
