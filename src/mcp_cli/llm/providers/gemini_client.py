from base64 import b64decode
import os
import logging
import json
import uuid
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

import google.genai as genai
from google.genai import types

from mcp_cli.llm.providers.base import BaseLLMClient 

load_dotenv()

class GeminiLLMClient(BaseLLMClient):
    def __init__(self, model="gemini-2.0-flash", api_key=None): 
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("The GOOGLE_API_KEY environment variable is not set.")

        self.client = genai.Client(api_key=self.api_key)
        logging.info(f"GeminiLLMClient initialized with model: {self.model}")

    def create_completion(self, messages: List[Dict], tools: Optional[List] = None) -> Dict[str, Any]:
        logging.warning(f"Creating completion with messages: {messages}, tools: {tools}")
        try:
            system_instruction_content = None
            gemini_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    if system_instruction_content is None:
                         system_instruction_content = msg["content"]
                elif msg["role"] in ["user", "assistant", "tool"]: 
                    role = "user" if msg["role"] == "user" else "model" if msg["role"] == "assistant" else "tool"
                    if role == "tool" and isinstance(msg.get("content"), str):
                         tool_call_id_for_response = msg.get("tool_call_id") 
                         tool_function_name = msg.get("name") 
                         if not tool_call_id_for_response and len(gemini_messages) > 0:
                              last_msg_content = gemini_messages[-1]
                              if last_msg_content.role == 'model' and last_msg_content.parts:
                                   for part in last_msg_content.parts:
                                       if hasattr(part, 'function_call'):
                                           tool_function_name = part.function_call.name
                                           logging.debug(f"Found preceding function call: {tool_function_name}")
                                           break 

                         if tool_function_name:
                            try:
                                function_response_data = json.loads(msg['content'])
                                
                                # a function that returns an image
                                for item in function_response_data:                                    
                                    if(item["type"] == "image"):
                                        function_response_part = types.Part.from_bytes(
                                            data=b64decode(item["data"]),
                                            mime_type=item["mimeType"]
                                        )
                                        gemini_messages.append(types.Content(role='user', parts=[function_response_part]))
                                    else:
                                        function_response_part = types.Part.from_function_response(
                                            name=tool_function_name, 
                                            response={"response":function_response_data}, 
                                        )
                                        gemini_messages.append(types.Content(role='tool', parts=[function_response_part]))
                                logging.debug(f"Formatted tool response for {tool_function_name}")
                            except json.JSONDecodeError:
                                logging.error(f"Failed to decode tool response content as JSON: {msg['content']}. Treating as text.")
                                gemini_messages.append(
                                     types.Content(role="user", parts=[types.Part.from_text(text=f"Tool response: {msg['content']}")]) # Fixed: Added text=
                                )
                            except Exception as tool_resp_err:
                                logging.error(f"Error formatting tool response part: {tool_resp_err}. Treating as text.")
                                gemini_messages.append(
                                     types.Content(role="user", parts=[types.Part.from_text(text=f"Tool response: {msg['content']}")]) # Fixed: Added text=
                                )
                         else:
                             # If we couldn't determine the function name, treat as text
                             logging.warning("Could not determine preceding function call name for tool response. Treating as text.")
                             gemini_messages.append(
                                 types.Content(role="user", parts=[types.Part.from_text(text=f"Tool response: {msg['content']}")]) # Fixed: Added text=
                             )

                    elif isinstance(msg.get("content"), str): 
                         gemini_messages.append(
                             types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]) 
                         )
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        logging.warning(msg["tool_calls"])
                        function_call_parts = []
                        for tc in msg["tool_calls"]:
                            func = tc.get("function", {})
                            name = func.get("name")
                            args_str = func.get("arguments", "{}")
                            if name:
                                try:
                                    args_dict = json.loads(args_str)
                                    function_call_parts.append(types.Part.from_function_call(name=name, args=args_dict))
                                except json.JSONDecodeError:
                                     logging.error(f"Failed to decode tool call arguments for {name}: {args_str}")
                                except Exception as fc_err:
                                     logging.error(f"Error creating function call part for {name}: {fc_err}")
                        if function_call_parts:
                             gemini_messages.append(types.Content(role="model", parts=function_call_parts))
                        elif msg.get("content"): # If tool calls failed but there's text content
                             gemini_messages.append(types.Content(role="model", parts=[types.Part.from_text(text=msg["content"])]))

                    else:
                         # Skip messages with non-string content if not a handled tool call/response
                         if msg.get("content") is not None:
                             logging.warning(f"Skipping message part with non-string/non-tool content: {msg}")


                else:
                    logging.warning(f"Unsupported role '{msg['role']}' found in messages.")


            generation_config_args = {}
            if system_instruction_content:
                generation_config_args["system_instruction"] = system_instruction_content

            tools_list = []
            tool_config_obj = None
            if tools:
                function_declarations = []
                for tool in tools:
                    if tool.get("type") == "function":
                        function = tool.get("function", {})
                        name = function.get("name")
                        description = function.get("description")
                        parameters = function.get("parameters")

                        if name and parameters is not None:
                            try:
                                # Parameters should be a dict representing a JSON schema
                                schema_parameters = types.Schema(**parameters)
                                function_declarations.append(
                                    types.FunctionDeclaration(
                                        name=name,
                                        description=description,
                                        parameters=schema_parameters
                                    )
                                )
                            except Exception as schema_error:
                                logging.error(f"Error creating Schema for tool '{name}': {schema_error}. Parameters: {parameters}")
                                raise ValueError(f"Invalid parameters schema for tool '{name}'.") from schema_error
                        else:
                             logging.warning(f"Skipping tool due to missing name, description, or parameters: {function}")

                if function_declarations:
                    tools_list = [types.Tool(function_declarations=function_declarations)]
                    tool_config_obj = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode=types.FunctionCallingConfigMode.AUTO 
                        )
                    )

            # Create the GenerationConfig object if args exist
            final_generation_config = types.GenerateContentConfig(**generation_config_args) if generation_config_args else None

            logging.debug(f"Calling generate_content with model='{self.model}', contents={gemini_messages}, generation_config={final_generation_config}, tools={tools_list}, tool_config={tool_config_obj}")

            # Generate response using SDK methods and parameters
            response = self.client.models.generate_content(
                model=self.model,
                contents=gemini_messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction_content,
                    tools=tools_list if tools_list else None,
                    tool_config=tool_config_obj
                ),
            )

            logging.debug(f"Received response from Gemini: {response}")

            # Extract the text response safely
            # The primary text might be within the first candidate's content parts
            main_response = ""
            tool_calls_list = [] # Renamed to avoid conflict

            if response.candidates:
                first_candidate = response.candidates[0]
                if first_candidate.content and first_candidate.content.parts:
                    for part in first_candidate.content.parts:
                        if hasattr(part, "text"):
                            if part.text:
                                main_response += part.text
                        elif hasattr(part, "function_call"):
                            # Process function calls if present in parts
                            fc = part.function_call
                            call_id = f"call_{uuid.uuid4().hex[:12]}" # Generate unique ID
                            fc_name = fc.name if hasattr(fc, "name") else "unknown_function"
                            # fc.args is a google.protobuf.struct_pb2.Struct (acts like dict)
                            fc_args = fc.args if hasattr(fc, "args") else {}

                            try:
                                # Convert StructProxy to dict, then dump to JSON string
                                arguments_str = json.dumps(dict(fc_args))
                            except (TypeError, ValueError) as json_err:
                                logging.error(f"Error serializing function call args for {fc_name}: {json_err}. Args: {fc_args}")
                                arguments_str = "{}"

                            tool_calls_list.append({
                                "id": call_id,
                                "type": "function", # Add type for consistency
                                "function": {
                                    "name": fc_name,
                                    "arguments": arguments_str,
                                },
                            })

            if not tool_calls_list and hasattr(response, "function_calls") and response.function_calls:
                logging.debug(f"Processing response.function_calls: {response.function_calls}")
                # tool_calls_list = [] # Reset only if prioritizing this attribute
                for fc in response.function_calls:
                    call_id = f"call_{uuid.uuid4().hex[:12]}" # Generate unique ID
                    fc_name = fc.name if hasattr(fc, "name") else "unknown_function"
                    fc_args = fc.args if hasattr(fc, "args") else {}

                    try:
                        # Arguments are already a dict-like StructProxy, convert to JSON string
                        arguments_str = json.dumps(dict(fc_args)) # Ensure it's a standard dict before dumping
                    except (TypeError, ValueError) as json_err:
                        logging.error(f"Error serializing function call args for {fc_name}: {json_err}. Args: {fc_args}")
                        arguments_str = "{}"

                    tool_calls_list.append({
                        "id": call_id,
                        "type": "function", # Add type for consistency
                        "function": {
                            "name": fc_name,
                            "arguments": arguments_str,
                        },
                    })


            # Check for blocked response (after checking candidates)
            if not response.candidates and hasattr(response, "prompt_feedback") and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                block_message = getattr(response.prompt_feedback, 'block_reason_message', '') # Use getattr for safety
                logging.warning(f"Request was blocked. Reason: {block_reason}, Message: {block_message}")
                # Return empty response or raise an error, depending on desired behavior
                return {
                    "response": f"Blocked due to {block_reason}. {block_message}",
                    "tool_calls": []
                }
                # Alternatively: raise ValueError(f"Blocked due to {block_reason}. {block_message}")

            # Return the standardized response format
            result = {
                "response": main_response.strip(),
                "tool_calls": tool_calls_list
            }
            logging.debug(f"Returning standardized result: {result}")
            return result

        except Exception as e:
            logging.error(f"Gemini API Error: {str(e)}", exc_info=True) # Log traceback
            raise ValueError(f"Gemini processing error: {str(e)}") from e

