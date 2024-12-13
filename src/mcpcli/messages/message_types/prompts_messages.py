# messages/message_types/prompts_messages.py
from typing import Any, Dict, Optional
from mcpcli.messages.message_types.incrementing_id_message import IncrementingIDMessage
from mcpcli.messages.message_types.json_rpc_message import JSONRPCMessage
from mcpcli.messages.message_types.prompts_models import PromptsGetParams

class PromptsListMessage(IncrementingIDMessage):
    def __init__(self, start_id: int = None, **kwargs):
        super().__init__(prefix="prompts-list", method="prompts/list", start_id=start_id, **kwargs)


class PromptsGetMessage(IncrementingIDMessage):
    def __init__(self, name: str, arguments: Optional[Dict[str, Any]] = None, start_id: int = None, **kwargs):
        # Validate params using PromptsGetParams
        params_model = PromptsGetParams(name=name, arguments=arguments or {})
        super().__init__(
            prefix="prompts-get",
            method="prompts/get",
            start_id=start_id,
            params=params_model.model_dump(),
            **kwargs
        )


class PromptsListChangedMessage(JSONRPCMessage):
    def __init__(self, **kwargs):
        super().__init__(method="notifications/prompts/list_changed", id=None, **kwargs)
