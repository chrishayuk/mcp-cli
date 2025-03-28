# src/llm/providers/bedrock_client.py
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

# Base client
from llm.providers.base import BaseLLMClient

class BedrockLLMClient(BaseLLMClient):
    def __init__(self, model: str = "claude-3-7-sonnet", region: str = "eu-central-1", 
                 model_id: str = None, arn: str = None, **kwargs):
        self.model = model
        self.region = region
        self.model_id = model_id
        self.arn = arn
        
        # Load config if needed
        if not self.model_id or not self.arn:
            self._load_model_config()
            
        # Initialize AWS Bedrock client
        try:
            # Import boto3 only when needed to avoid issues if it's not installed
            import boto3
            self.client = boto3.client('bedrock-runtime', region_name=self.region)
        except ImportError:
            raise ValueError("boto3 is not installed. Please install it using 'pip install boto3' or 'uv sync --reinstall'")
        except Exception as e:
            logging.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
            raise ValueError(f"AWS Bedrock client initialization error: {str(e)}")
    
    def _load_model_config(self):
        """Load model configuration from config.json if available"""
        try:
            with open("server_config.json", "r") as f:
                config = json.load(f)
                
            bedrock_config = config.get("llmProviders", {}).get("bedrock", {})
            model_config = bedrock_config.get("models", {}).get(self.model, {})
            
            if not self.model_id:
                self.model_id = model_config.get("modelId")
            if not self.region:
                self.region = bedrock_config.get("region", "eu-central-1")
            if not self.arn:
                self.arn = model_config.get("arn")
                
        except Exception as e:
            logging.warning(f"Failed to load Bedrock model config: {str(e)}")
    
    def create_completion(self, messages: List[Dict], tools: List = None) -> Dict[str, Any]:
        try:
            # Ensure boto3 is imported
            import boto3
            
            # Format messages for the Claude model
            formatted_messages = self._format_messages(messages)
            
            # Format tools for Claude if provided
            if tools and len(tools) > 0:
                formatted_tools = self._format_tools(tools)
            else:
                formatted_tools = None
            
            # Prepare the request body
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": formatted_messages
            }
            
            # Add tools if available
            if formatted_tools:
                request_body["tools"] = formatted_tools
            
            # Invoke the model
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Process the response
            response_body = json.loads(response['body'].read().decode('utf-8'))
            
            # Extract content and tool calls
            content = response_body.get('content', [{"type": "text", "text": "No response"}])
            text_content = ""
            tool_calls = []
            
            # Process different content blocks
            for block in content:
                if block.get("type") == "text":
                    text_content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_use = block.get("tool_use", {})
                    tool_call = {
                        "id": f"call_{tool_use.get('name')}_{str(uuid.uuid4())[:8]}",
                        "type": "function",
                        "function": {
                            "name": tool_use.get("name"),
                            "arguments": json.dumps(tool_use.get("input", {}))
                        }
                    }
                    tool_calls.append(tool_call)
            
            # Return standardized response format
            return {
                "response": text_content,
                "tool_calls": tool_calls,
            }
            
        except Exception as e:
            logging.error(f"Bedrock API Error: {str(e)}")
            raise ValueError(f"Bedrock API Error: {str(e)}")
    
    def _format_messages(self, messages: List[Dict]) -> List[Dict]:
        """Format messages for Bedrock API"""
        formatted_messages = []
        
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            
            if role == "system":
                # System messages handled differently in Claude
                formatted_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"<system>\n{content}\n</system>"}]
                })
                # Add an assistant response to complete the turn
                formatted_messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I'll help you with that."}]
                })
            elif role == "user":
                formatted_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": content}]
                })
            elif role == "assistant":
                # Handle potential tool calls in assistant messages
                formatted_content = []
                
                # Add text content
                if content:
                    formatted_content.append({"type": "text", "text": content})
                
                formatted_messages.append({
                    "role": "assistant",
                    "content": formatted_content
                })
        
        # Ensure the conversation has an even number of turns
        if len(formatted_messages) % 2 == 1:
            formatted_messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": "I'll help you with that."}]
            })
        
        return formatted_messages
    
    def _format_tools(self, tools: List) -> List[Dict]:
        """Format tools for Bedrock API"""
        formatted_tools = []
        
        for tool in tools:
            formatted_tool = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            }
            
            # Add parameters to the schema
            parameters = tool.get("parameters", {}).get("properties", {})
            required = tool.get("parameters", {}).get("required", [])
            
            for param_name, param_details in parameters.items():
                formatted_tool["input_schema"]["properties"][param_name] = {
                    "type": param_details.get("type", "string"),
                    "description": param_details.get("description", "")
                }
            
            # Add required parameters
            if required:
                formatted_tool["input_schema"]["required"] = required
            
            formatted_tools.append(formatted_tool)
        
        return formatted_tools