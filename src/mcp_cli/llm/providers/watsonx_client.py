# src/llm/providers/watsonx_client.py
import os
import json
import uuid
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# base
from mcp_cli.llm.providers.base import BaseLLMClient

# Load environment variables
load_dotenv()

class WatsonxLLMClient(BaseLLMClient):
    def __init__(self, model: str = "ibm/granite-3-8b-instruct"):
        self.model = model

        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        api_key = os.getenv("WATSONX_API_KEY")
        endpoint = os.getenv("WATSONX_ENDPOINT_URL")

        if not all([self.project_id, api_key, endpoint]):
            raise ValueError("Missing watsonx credentials in .env")

        credentials = Credentials(url=endpoint, api_key=api_key)
        self.client = ModelInference(
            model_id=self.model,
            credentials=credentials,
            project_id=self.project_id,
        )

    def create_completion(self, messages: List[Dict], tools: List = None) -> Dict[str, Any]:
        try:

            # Call chat() with tool support
            response = self.client.chat(
                messages=messages,
                tools=tools,
                tool_choice_option="auto"
            )
 
            return {
                "response": response['choices'][0]['message']['content'] if "content" in response['choices'][0]['message'] else "",
                "tool_calls": response['choices'][0]['message']["tool_calls"] if "tool_calls" in response['choices'][0]['message'] else [],
            }

        except Exception as e:
            logging.exception("Watsonx API Error")
            raise ValueError(f"Watsonx API Error: {e}")
