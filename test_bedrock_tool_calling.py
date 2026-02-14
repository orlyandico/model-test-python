#!/usr/bin/env python3
"""
Minimal test to demonstrate tool calling issue with zai.glm-4.7 on Bedrock.
Works with moonshotai.kimi-k2.5 and us.amazon.nova-micro-v1:0 but fails with zai.glm-4.7.
"""
import os
import boto3
import json

# Initialize Bedrock client
client = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

# Define a simple tool
tools = [{
    "toolSpec": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
}]

def test_model(model_id, model_name):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} ({model_id})")
    print(f"{'='*60}")
    
    try:
        # Round 1: Initial request
        print("\n--- Round 1: Initial request ---")
        response = client.converse(
            modelId=model_id,
            messages=[{
                "role": "user",
                "content": [{"text": "What's the weather in Paris?"}]
            }],
            toolConfig={"tools": tools}
        )
        
        output = response["output"]["message"]
        print(f"Response: {json.dumps(output, indent=2)}")
        
        # Check if tool was called
        tool_use = None
        for item in output["content"]:
            if "toolUse" in item:
                tool_use = item["toolUse"]
                break
        
        if not tool_use:
            print("❌ No tool call made")
            return
        
        print(f"✓ Tool called: {tool_use['name']} with args {tool_use['input']}")
        
        # Round 2: Send tool result back
        print("\n--- Round 2: Tool result ---")
        response2 = client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "What's the weather in Paris?"}]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "toolUse": {
                            "toolUseId": tool_use["toolUseId"],
                            "name": tool_use["name"],
                            "input": tool_use["input"]
                        }
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": [{"text": "Sunny, 22°C"}]
                        }
                    }]
                }
            ],
            toolConfig={"tools": tools}
        )
        
        output2 = response2["output"]["message"]
        final_text = ""
        for item in output2["content"]:
            if "text" in item:
                final_text = item["text"]
                break
        
        print(f"✓ Final response: {final_text}")
        print(f"✅ SUCCESS")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")

# Test models
test_model("moonshotai.kimi-k2.5", "Moonshot Kimi K2.5")
test_model("us.amazon.nova-micro-v1:0", "Amazon Nova Micro")
test_model("deepseek.v3.2", "DeepSeek V3.2")
test_model("zai.glm-4.7", "Z.AI GLM 4.7")
test_model("zai.glm-4.7-flash", "Z.AI GLM 4.7 Flash")
