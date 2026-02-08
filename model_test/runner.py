import json
import time
import os
from openai import OpenAI
import boto3
from .models import AgentResponse, ToolCall
from .tools import TOOLS, CartService, execute_tool


def _is_api_error(exc: Exception) -> bool:
    """Check if an exception is a fatal API/HTTP error that should abort the run.

    Covers OpenAI SDK errors (used by Ollama, llama.cpp, Vertex MaaS),
    boto3/botocore errors (Bedrock), and Google API errors (Vertex AI).
    """
    # OpenAI SDK errors (APIStatusError covers 4xx/5xx, APIConnectionError
    # covers network failures)
    try:
        from openai import APIStatusError, APIConnectionError
        if isinstance(exc, (APIStatusError, APIConnectionError)):
            return True
    except ImportError:
        pass

    # boto3 / botocore errors (Bedrock)
    try:
        from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
        if isinstance(exc, (ClientError, EndpointConnectionError, NoCredentialsError)):
            return True
    except ImportError:
        pass

    # Google Cloud errors (Vertex AI)
    try:
        from google.api_core.exceptions import GoogleAPIError
        if isinstance(exc, GoogleAPIError):
            return True
    except ImportError:
        pass

    return False


# LLM Judge configuration
LLM_JUDGE_REGION = "us-west-2"
LLM_JUDGE_MODEL_ID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"


def evaluate_with_llm_judge(
    prompt: str,
    actual_calls: list[ToolCall],
    expected_variants: list,
) -> tuple[bool, str, str]:
    """
    Use Claude Sonnet 4.5 as an LLM judge to evaluate tool selection.

    NOT ALL-OR-NOTHING: We focus on whether the tool sequence makes sense
    for the user's intent, not exact parameter matches.

    Args:
        prompt: The original user prompt
        actual_calls: List of actual tool calls made by the model
        expected_variants: List of expected tool call variants

    Returns:
        Tuple of (success, matched_variant_name, reasoning)
    """
    # Format actual calls for the judge
    actual_formatted = []
    for call in actual_calls:
        actual_formatted.append({
            "tool_name": call.tool_name,
            "arguments": call.arguments
        })

    # Format expected variants for the judge
    expected_formatted = []
    for variant in expected_variants:
        variant_tools = []
        for tool in variant.tools:
            variant_tools.append({
                "tool_name": tool.name,
                "arguments": tool.arguments
            })
        expected_formatted.append({
            "variant_name": variant.name,
            "description": variant.description,
            "tools": variant_tools
        })

    # Build the judge prompt
    judge_prompt = f"""You are evaluating whether an LLM's tool calls appropriately address a user's request.

USER'S ORIGINAL REQUEST:
{prompt}

ACTUAL TOOL CALLS MADE BY THE MODEL:
{json.dumps(actual_formatted, indent=2)}

EXPECTED TOOL CALL PATTERNS (any of these would be acceptable):
{json.dumps(expected_formatted, indent=2)}

EVALUATION CRITERIA:
1. Tool Selection: Did the model call the right tool(s) for the user's intent?
2. Sequence Logic: Does the sequence of tool calls make sense for the task?
3. Parameter Reasonableness: Are the arguments reasonable for the intent? (Exact matches NOT required - e.g., "blue shirt" vs "Blue T-Shirt" are equivalent if they serve the same purpose)

IMPORTANT:
- Focus on whether the tool calls MAKE SENSE for the user's intent
- Do NOT require exact parameter matches (e.g., product names don't need to match exactly)
- Give credit for partial correctness - if most tools are right, that's good
- Consider the SPIRIT of the request, not just literal interpretation

Respond with a JSON object:
{{
    "success": true/false,
    "matched_variant": "variant name if matched, or empty string",
    "reasoning": "brief explanation of your evaluation"
}}

Only output the JSON object, nothing else."""

    try:
        # Create Bedrock client for the judge
        judge_client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=LLM_JUDGE_REGION
        )

        # Call Claude Sonnet 4.5 as judge
        response = judge_client.converse(
            modelId=LLM_JUDGE_MODEL_ID,
            messages=[{
                "role": "user",
                "content": [{"text": judge_prompt}]
            }],
            inferenceConfig={
                "temperature": 0
            }
        )

        # Parse the response
        output_text = response["output"]["message"]["content"][0]["text"]

        # Extract JSON from response (handle potential markdown code blocks)
        if "```json" in output_text:
            output_text = output_text.split("```json")[1].split("```")[0]
        elif "```" in output_text:
            output_text = output_text.split("```")[1].split("```")[0]

        result = json.loads(output_text.strip())

        return (
            result.get("success", False),
            result.get("matched_variant", ""),
            result.get("reasoning", "")
        )

    except Exception as e:
        print(f"  ⚠️  LLM Judge error: {e}")
        # Fallback to basic matching if judge fails
        return False, "", f"Judge error: {str(e)}"


class BedrockClient:
    """Bedrock client for Converse API with tool calling."""
    
    def __init__(self, model_id: str):
        self.client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        self.model_id = model_id
    
    def _convert_tools(self):
        """Convert OpenAI tool format to Bedrock format."""
        bedrock_tools = []
        for tool in TOOLS:
            func = tool["function"]
            bedrock_tools.append({
                "toolSpec": {
                    "name": func["name"],
                    "description": func["description"],
                    "inputSchema": {
                        "json": func["parameters"]
                    }
                }
            })
        return bedrock_tools
    
    def _convert_messages(self, messages):
        """Convert OpenAI messages to Bedrock format."""
        bedrock_messages = []
        system_prompts = []
        i = 0

        while i < len(messages):
            msg = messages[i]

            # Handle both dict and object formats
            if isinstance(msg, dict):
                role = msg["role"]
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", None)
                content = getattr(msg, "content", "")

            if role == "system":
                system_prompts.append({"text": content})
                i += 1
            elif role == "user":
                bedrock_messages.append({
                    "role": "user",
                    "content": [{"text": content}]
                })
                i += 1
            elif role == "assistant":
                bedrock_content = []
                # Check for text content
                if content:
                    bedrock_content.append({"text": content})
                # Check for tool calls
                tool_calls = getattr(msg, "tool_calls", None) if not isinstance(msg, dict) else msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        bedrock_content.append({
                            "toolUse": {
                                "toolUseId": tc.id,
                                "name": tc.function.name,
                                "input": json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                            }
                        })
                bedrock_messages.append({"role": "assistant", "content": bedrock_content})
                i += 1
            elif role == "tool":
                # Accumulate all consecutive tool results into one user message
                tool_results = []
                while i < len(messages):
                    tool_msg = messages[i]
                    tool_role = tool_msg["role"] if isinstance(tool_msg, dict) else getattr(tool_msg, "role", None)

                    if tool_role != "tool":
                        break

                    if isinstance(tool_msg, dict):
                        tool_call_id = tool_msg["tool_call_id"]
                        result_content = tool_msg["content"]
                    else:
                        tool_call_id = getattr(tool_msg, "tool_call_id", None)
                        result_content = getattr(tool_msg, "content", "")

                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_call_id,
                            "content": [{"text": result_content}]
                        }
                    })
                    i += 1

                # Add all tool results as a single user message
                bedrock_messages.append({
                    "role": "user",
                    "content": tool_results
                })
            else:
                i += 1

        return system_prompts, bedrock_messages
    
    def create_completion(self, messages):
        """Create completion using Bedrock Converse API."""
        system_prompts, bedrock_messages = self._convert_messages(messages)
        
        kwargs = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "toolConfig": {"tools": self._convert_tools()}
        }
        
        if system_prompts:
            kwargs["system"] = system_prompts
        
        response = self.client.converse(**kwargs)
        
        # Convert response to OpenAI-like format
        output = response["output"]["message"]

        class Message:
            def __init__(self, content, tool_calls):
                self.role = "assistant"
                self.content = content
                self.tool_calls = tool_calls

        class ToolCall:
            def __init__(self, id, name, arguments):
                self.id = id
                self.function = type('obj', (object,), {
                    'name': name,
                    'arguments': json.dumps(arguments)
                })()

        class Choice:
            def __init__(self, message):
                self.message = message

        class Response:
            def __init__(self, choices):
                self.choices = choices
        
        content = ""
        tool_calls = []
        
        for item in output["content"]:
            if "text" in item:
                content = item["text"]
            elif "toolUse" in item:
                tool_use = item["toolUse"]
                tool_calls.append(ToolCall(
                    id=tool_use["toolUseId"],
                    name=tool_use["name"],
                    arguments=tool_use["input"]
                ))
        
        message = Message(content, tool_calls if tool_calls else None)
        return Response([Choice(message)])


class VertexAIClient:
    """Vertex AI client for Gemini models with tool calling."""

    def __init__(self, model_id: str):
        import vertexai

        project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        vertexai.init(project=project, location=location)

        self.model_id = model_id
        self._tools = self._convert_tools()
        self._location = location
        self._project = project

    def _convert_tools(self):
        """Convert OpenAI tool format to Vertex AI format."""
        from vertexai.generative_models import FunctionDeclaration, Tool as VertexTool

        declarations = []
        for tool in TOOLS:
            func = tool["function"]
            params = func.get("parameters", {"type": "object", "properties": {}})
            declarations.append(FunctionDeclaration(
                name=func["name"],
                description=func["description"],
                parameters=params
            ))
        return [VertexTool(function_declarations=declarations)]

    def _convert_messages(self, messages):
        """Convert OpenAI message format to Vertex AI format."""
        from vertexai.generative_models import Content, Part

        system_instruction = None
        contents = []
        tool_id_to_name = {}

        i = 0
        while i < len(messages):
            msg = messages[i]
            if isinstance(msg, dict):
                role = msg["role"]
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", None)
                content = getattr(msg, "content", "")

            if role == "system":
                system_instruction = content
                i += 1
            elif role == "user":
                contents.append(Content(role="user", parts=[Part.from_text(content)]))
                i += 1
            elif role == "assistant":
                parts = []
                if content:
                    parts.append(Part.from_text(content))
                tool_calls = getattr(msg, "tool_calls", None) if not isinstance(msg, dict) else msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func_name = tc.function.name
                        tool_id_to_name[tc.id] = func_name
                        args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                        parts.append(Part.from_dict({"function_call": {"name": func_name, "args": args}}))
                if parts:
                    contents.append(Content(role="model", parts=parts))
                i += 1
            elif role == "tool":
                # Accumulate consecutive tool results into one user message
                parts = []
                while i < len(messages):
                    tool_msg = messages[i]
                    tool_role = tool_msg["role"] if isinstance(tool_msg, dict) else getattr(tool_msg, "role", None)
                    if tool_role != "tool":
                        break

                    tool_call_id = tool_msg["tool_call_id"] if isinstance(tool_msg, dict) else getattr(tool_msg, "tool_call_id", None)
                    result_content = tool_msg["content"] if isinstance(tool_msg, dict) else getattr(tool_msg, "content", "")

                    func_name = tool_id_to_name.get(tool_call_id, "unknown")
                    parts.append(Part.from_function_response(
                        name=func_name,
                        response={"result": result_content}
                    ))
                    i += 1

                contents.append(Content(role="user", parts=parts))
            else:
                i += 1

        return system_instruction, contents

    def create_completion(self, messages):
        """Create completion using Vertex AI Gemini API."""
        import uuid
        from vertexai.generative_models import GenerativeModel

        system_instruction, contents = self._convert_messages(messages)

        if system_instruction:
            model = GenerativeModel(self.model_id, system_instruction=system_instruction)
        else:
            model = GenerativeModel(self.model_id)

        response = model.generate_content(contents=contents, tools=self._tools)

        # Convert response to OpenAI-like format
        class Message:
            def __init__(self, content, tool_calls):
                self.role = "assistant"
                self.content = content
                self.tool_calls = tool_calls

        class ToolCallObj:
            def __init__(self, id, name, arguments):
                self.id = id
                self.function = type('obj', (object,), {
                    'name': name,
                    'arguments': json.dumps(arguments)
                })()

        class Choice:
            def __init__(self, message):
                self.message = message

        class Response:
            def __init__(self, choices):
                self.choices = choices

        content_text = ""
        tool_calls = []

        for part in response.candidates[0].content.parts:
            if part.function_call and part.function_call.name:
                fc = part.function_call
                tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                args = dict(fc.args) if fc.args else {}
                tool_calls.append(ToolCallObj(
                    id=tool_call_id,
                    name=fc.name,
                    arguments=args
                ))
            elif part.text:
                content_text += part.text

        message = Message(content_text, tool_calls if tool_calls else None)
        return Response([Choice(message)])


class TestRunner:
    def __init__(self, api_key: str, base_url: str, model: str, host: str = "localhost"):
        self.model = model or ""
        self.backend_type = None
        self.actual_base_url = base_url
        self.is_bedrock = False
        self.client = None

        # Check if using Bedrock
        if self.model.startswith("bedrock/"):
            bedrock_model = self.model.replace("bedrock/", "")
            self.client = BedrockClient(bedrock_model)
            self.is_bedrock = True
            self.backend_type = "bedrock"
            self.model = bedrock_model
        # Check if using Vertex AI (native Gemini models)
        elif self.model.startswith("vertex/"):
            vertex_model = self.model.replace("vertex/", "")
            self.client = VertexAIClient(vertex_model)
            self.is_bedrock = False
            self.backend_type = "vertex"
            self.model = vertex_model
            self.actual_base_url = "Vertex AI API"
        # Check if using Vertex AI Model Garden MaaS (OpenAI-compatible)
        elif self.model.startswith("vertex-maas/"):
            import google.auth
            import google.auth.transport.requests

            maas_model = self.model.replace("vertex-maas/", "")
            credentials, default_project = google.auth.default()
            credentials.refresh(google.auth.transport.requests.Request())

            project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT") or default_project
            location = os.getenv("VERTEX_MAAS_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION", "global")
            if location == "global":
                maas_url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi"
            else:
                maas_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi"

            self.client = OpenAI(api_key=credentials.token, base_url=maas_url)
            self.model = maas_model
            self.is_bedrock = False
            self.backend_type = "vertex-maas"
            self.actual_base_url = maas_url
        # Check if using llama.cpp server
        elif self.model.startswith("llama.cpp/"):
            actual_model = self.model.replace("llama.cpp/", "")
            llama_cpp_url = f"http://{host}:8080/v1"
            self.client = OpenAI(api_key=api_key or "not-needed", base_url=llama_cpp_url)
            self.model = actual_model
            self.backend_type = "llama.cpp"
            self.actual_base_url = llama_cpp_url
        # Check if using ollama with explicit prefix
        elif self.model.startswith("ollama/"):
            actual_model = self.model.replace("ollama/", "")
            ollama_url = f"http://{host}:11434/v1"
            self.client = OpenAI(api_key=api_key or "ollama", base_url=ollama_url)
            self.model = actual_model
            self.backend_type = "ollama"
            self.actual_base_url = ollama_url
        else:
            # No valid prefix provided
            self.backend_type = None
    
    def run_agent_test(self, test_case) -> tuple[AgentResponse | None, float, str]:
        """Run a single agent test with up to 10 rounds."""
        cart = CartService()

        # Initialise cart if needed
        if test_case.initial_cart_state:
            for item in test_case.initial_cart_state.items:
                cart.add_to_cart(item.product_name, item.quantity)

        messages = [
            {"role": "system", "content": "You are a helpful shopping assistant. Use the provided tools to help users."},
            {"role": "user", "content": test_case.prompt}
        ]

        all_tool_calls = []
        llm_requests = 0
        llm_total_time = 0.0
        max_rounds = 10

        print(f"\n{'='*60}")
        print(f"Test: {test_case.name}")
        print(f"Prompt: {test_case.prompt}")
        print(f"{'='*60}")

        try:
            for round_num in range(max_rounds):
                print(f"\n--- Round {round_num + 1}/10 ---")
                start = time.time()

                if self.backend_type in ("bedrock", "vertex"):
                    response = self.client.create_completion(messages)
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=TOOLS,
                    )

                llm_time = time.time() - start
                llm_requests += 1
                llm_total_time += llm_time

                message = response.choices[0].message
                print(f"LLM response time: {llm_time:.2f}s")

                # No tool calls - done
                if not message.tool_calls:
                    final_msg = message.content or ""
                    print(f"✓ Final message: {final_msg}")
                    print(f"✓ Test completed in {llm_requests} rounds")
                    agent_response = AgentResponse(
                        tool_calls=all_tool_calls,
                        llm_requests=llm_requests,
                        llm_total_time=llm_total_time,
                        final_message=final_msg
                    )
                    return agent_response, llm_total_time, ""

                # Execute tool calls
                print(f"Tool calls requested: {len(message.tool_calls)}")
                messages.append(message)

                for idx, tool_call in enumerate(message.tool_calls, 1):
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments

                    print(f"  [{idx}] {tool_name}({json.dumps(arguments)})")

                    all_tool_calls.append(ToolCall(tool_name=tool_name, arguments=arguments))

                    result = execute_tool(tool_name, arguments, cart)
                    print(f"      → Result: {result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            
            # Max rounds exceeded
            print(f"\n⚠️  WARNING: Max rounds ({max_rounds}) exceeded - test failed")
            print(f"   Total tool calls made: {len(all_tool_calls)}")
            agent_response = AgentResponse(
                tool_calls=all_tool_calls,
                llm_requests=llm_requests,
                llm_total_time=llm_total_time
            )
            return agent_response, llm_total_time, "Max rounds exceeded"

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            # Abort on API/HTTP errors — these are fatal and won't resolve
            # by retrying the next test case.
            if _is_api_error(e):
                print(f"\n❌ FATAL API ERROR: {str(e)}")
                raise SystemExit(1) from e
            print(f"\n❌ ERROR: {str(e)}")
            return None, llm_total_time, str(e)
    
    def _brittle_match(
        self,
        actual_calls: list[ToolCall],
        expected_variants: list
    ) -> str:
        """Try exact matching of tool names and arguments.

        Returns the matched variant name, or empty string if no match.
        """
        for variant in expected_variants:
            if len(actual_calls) != len(variant.tools):
                continue

            match = True
            for actual, expected in zip(actual_calls, variant.tools):
                if actual.tool_name != expected.name:
                    match = False
                    break

                # If expected.arguments is None or empty, accept any arguments (wildcard)
                if not expected.arguments:
                    continue

                # Check required arguments (allow extra arguments)
                for key, value in expected.arguments.items():
                    if key not in actual.arguments:
                        match = False
                        break
                    # Allow flexible matching for string values
                    if isinstance(value, str) and isinstance(actual.arguments.get(key), str):
                        expected_val = value.lower()
                        actual_val = actual.arguments[key].lower()
                        if expected_val not in actual_val and actual_val not in expected_val:
                            match = False
                            break

                if not match:
                    break

            if match:
                return variant.name

        return ""

    def match_tool_path(
        self,
        actual_calls: list[ToolCall],
        expected_variants: list,
        prompt: str = ""
    ) -> str:
        """Check if actual tool calls match any expected variant.

        First tries brittle (exact) matching for speed and reliability.
        Falls back to LLM-as-judge for semantic evaluation if brittle match fails.

        Args:
            actual_calls: The actual tool calls made by the model
            expected_variants: List of acceptable tool call patterns
            prompt: The original user prompt (for context in LLM evaluation)

        Returns:
            The name of the matched variant, or empty string if no match
        """
        # Check if any variant expects tools to be called
        any_variant_expects_tools = any(
            len(variant.tools) > 0 for variant in expected_variants
        ) if expected_variants else False

        # Handle case: no tools expected and none called - clear success
        if not any_variant_expects_tools and not actual_calls:
            if expected_variants:
                return expected_variants[0].name
            return "no_tools_expected"

        # Handle case: tools expected but none called - clear failure
        if any_variant_expects_tools and not actual_calls:
            return ""

        # Step 1: Try brittle (exact) matching first - fast and reliable
        if any_variant_expects_tools:
            brittle_result = self._brittle_match(actual_calls, expected_variants)
            if brittle_result:
                return brittle_result

        # Step 2: Fall back to LLM judge for semantic evaluation
        # This handles:
        # - Brittle match failed but tools might still be semantically correct
        # - No tools expected but model called tools (might be reasonable, e.g., searching for AI books when asked about AI)
        if any_variant_expects_tools:
            print("  🤖 Brittle match failed, evaluating with LLM judge...")
        else:
            print("  🤖 No tools expected but tools were called, evaluating with LLM judge...")

        success, matched_variant, reasoning = evaluate_with_llm_judge(
            prompt=prompt,
            actual_calls=actual_calls,
            expected_variants=expected_variants
        )

        if reasoning:
            print(f"  📝 Judge reasoning: {reasoning}")

        if success:
            # For no-tools-expected case, return a sensible name
            if not any_variant_expects_tools:
                return "acceptable_tool_use"
            return matched_variant if matched_variant else expected_variants[0].name

        return ""
