#!/usr/bin/env python3
"""
Test script to diagnose Claude's tool calling issue.
Uses the same prompt building and parsing as main.py.
"""

import asyncio
import json
import os
import sys
import time
from typing import List, Dict, Any, Tuple

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worker import PlaywrightWorker
# Import the prompt building and parsing functions from main.py
from main import _build_conversation_for_claude, _parse_claude_response, Message, _needs_tool_call

# Correction prompt for retry - same as in main.py
_CORRECTION_PROMPT = (
    "Your previous response did not contain a <tool_call> block. "
    "You MUST respond with ONLY a <tool_call> block — no explanation, "
    "no preamble, no trailing text.\n\n"
    "Correct format:\n"
    "<tool_call>{\"tool\": \"TOOL_NAME\", \"input\": {...}}</tool_call>\n\n"
    "Now output the correct <tool_call>:"
)


class ClaudeToolCallTester:
    def __init__(self):
        self.worker = None
        self.test_results = []
        
    async def setup(self):
        """Initialize the worker"""
        print("Setting up worker...")
        self.worker = PlaywrightWorker(headless=False)  # Set to False to see the browser
        await self.worker.start()
        print("Worker ready")
        
    async def teardown(self):
        """Clean up worker"""
        if self.worker:
            await self.worker.stop()
            
    def has_tool_use(self, blocks: List[Dict]) -> Tuple[bool, str]:
        """
        Check if the parsed blocks contain a tool_use block.
        Returns (has_tool_use, reason)
        """
        tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
        if tool_blocks:
            return True, f"Found {len(tool_blocks)} tool_use block(s)"
        else:
            # Check if there are text blocks that might be conversational
            text_blocks = [b for b in blocks if b.get("type") == "text"]
            if text_blocks:
                # Check if the text looks like a conversational response
                first_text = text_blocks[0].get("text", "").strip()
                if first_text:
                    # Look for common conversational starters
                    conversational_starts = [
                        "I'll", "I will", "Let me", "Sure", "Okay", "Of course",
                        "I'd be happy", "I can", "Here's", "First", "To",
                        "I need", "I should", "I'm", "I am", "Great", "Sure thing"
                    ]
                    for starter in conversational_starts:
                        if first_text.startswith(starter):
                            return False, f"Conversational response: '{first_text[:50]}...'"
                    # If it's not obviously conversational, still consider it a failure
                    # because we expected a tool use
                    return False, f"Text response instead of tool use: '{first_text[:50]}...'"
            return False, "No tool_use blocks in parsed response"
    
    async def run_test(self, test_name: str, description: str, tools: List[Dict], history: List = None):
        """Run a single test case"""
        if history is None:
            history = []
            
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")
        print(f"Description: {description}")
        print(f"Tools: {[t['name'] for t in tools]}")
        print(f"History length: {len(history)} messages")
        print("-" * 60)

        # Build messages for the prompt with history
        messages = list(history)  # Copy history
        messages.append(Message(role="user", content=description))
        
        # Build the prompt using the same function as main.py
        prompt = _build_conversation_for_claude(messages, tools)
        
        print("Sending prompt to Claude...")
        print(f"Prompt:\\n{prompt}\\n---")
        start_time = time.time()
        
        try:
            response = await self.worker.query(prompt)
            elapsed = time.time() - start_time
            
            print(f"Response received in {elapsed:.1f}s")
            print(f"Response length: {len(response)} chars")
            print(f"\n--- RAW RESPONSE (first 300 chars) ---")
            print(response[:300] + ("..." if len(response) > 300 else ""))
            print("--- END RESPONSE ---\n")
            
            # Parse the response using the same function as main.py
            blocks = _parse_claude_response(response, tools)
            print(f"Parsed into {len(blocks)} blocks:")
            for i, block in enumerate(blocks):
                print(f" Block {i}: {block.get('type')} -> {str(block)[:100]}")
            
            has_tool, reason = self.has_tool_use(blocks)
        
            # FIX #3: AUTOMATIC RETRY ON MISSING TOOL CALL
            # Keep retrying until we get a tool call or hit max retries
            max_retries = 5
            retry_count = 0
            while tools and not has_tool and _needs_tool_call(messages) and retry_count < max_retries:
                retry_count += 1
                print(f"\n[RETRY {retry_count}/{max_retries}] No tool call in response — sending correction prompt")
                correction = prompt + "\n\n" + response + "\n\nUser: " + _CORRECTION_PROMPT + "\nAssistant: <tool_call>"
                response = await self.worker.query(correction)
                print(f"[RETRY {retry_count}] ← LLM response ({len(response)} chars): {response[:120]}{'...' if len(response) > 120 else ''}")
        
                blocks = _parse_claude_response(response, tools)
                tool_count = len([b for b in blocks if b.get("type") == "tool_use"])
                if tool_count > 0:
                    print(f"[RETRY {retry_count}] Succeeded: {tool_count} tool_use block(s)")
                    has_tool = True
                    reason = f"Found {tool_count} tool_use block(s) after {retry_count} retry(ies)"
                else:
                    print(f"[RETRY {retry_count}] Still no tool_use — will retry")
                    # Wait a bit between retries to avoid rate limiting
                    await asyncio.sleep(2)
            
            result = {
                "test_name": test_name,
                "description": description,
                "success": has_tool,
                "reason": reason,
                "response_preview": response[:200],
                "elapsed_time": elapsed,
                "num_blocks": len(blocks),
                "tool_blocks": len([b for b in blocks if b.get("type") == "tool_use"])
            }
            self.test_results.append(result)
            
            if has_tool:
                print(f"✅ SUCCESS: {reason}")
            else:
                print(f"❌ FAILED: {reason}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.test_results.append({
                "test_name": test_name,
                "description": description,
                "success": False,
                "reason": f"Exception: {e}",
                "response_preview": "",
                "elapsed_time": 0,
                "num_blocks": 0,
                "tool_blocks": 0
            })
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["success"])
        failed = total - passed
        
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success rate: {passed/total*100:.1f}%" if total > 0 else "N/A")
        
        if failed > 0:
            print(f"\n--- FAILED TESTS ---")
            for r in self.test_results:
                if not r["success"]:
                    print(f"\n• {r['test_name']}")
                    print(f"  Reason: {r['reason']}")
                    print(f"  Response preview: {r['response_preview'][:100]}...")
                    print(f"  Blocks: {r['num_blocks']} (tool: {r['tool_blocks']})")


async def main():
    # Check environment
    provider = os.getenv("TARGET_PROVIDER", "gemini").lower()
    if provider != "claude":
        print("WARNING: TARGET_PROVIDER is not set to 'claude'")
        print("Please run: export TARGET_PROVIDER=claude")
        print("Continuing anyway...\n")
    
    tester = ClaudeToolCallTester()
    
    try:
        await tester.setup()
        
        # Define test cases with conversation history
        test_cases = [
            {
                "name": "Simple Bash command",
                "description": "Run a command to list files in the current directory",
                "tools": [{
                    "name": "Bash",
                    "description": "Execute a bash command",
                    "input_schema": {
                        "properties": {
                            "command": {"type": "string"}
                        }
                    }
                }],
                "history": []
            },
            {
                "name": "Write file",
                "description": "Create a Python file named hello.py that prints Hello World",
                "tools": [{
                    "name": "Write",
                    "description": "Write content to a file",
                    "input_schema": {
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    }
                }],
                "history": []
            },
            {
                "name": "Read file",
                "description": "Read the contents of main.py",
                "tools": [{
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {
                        "properties": {
                            "file_path": {"type": "string"}
                        }
                    }
                }],
                "history": []
            },
            {
                "name": "Multi-step task with history",
                "description": "Create a simple FastAPI app with one endpoint",
                "tools": [
                    {
                        "name": "Write",
                        "description": "Write content to a file",
                        "input_schema": {
                            "properties": {
                                "file_path": {"type": "string"},
                                "content": {"type": "string"}
                            }
                        }
                    },
                    {
                        "name": "Bash",
                        "description": "Execute a bash command",
                        "input_schema": {
                            "properties": {
                                "command": {"type": "string"}
                            }
                        }
                    }
                ],
                "history": [
                    Message(role="user", content="I need to set up a Python project"),
                    Message(role="assistant", content="<tool_call>{\"tool\": \"Bash\", \"input\": {\"command\": \"mkdir -p myproject\"}}</tool_call>"),
                    Message(role="tool", content="[Tool result]: Directory created successfully"),
                ]
            },
            {
                "name": "Edit existing file",
                "description": "Overwrite main.py with a new FastAPI app that has a /hello endpoint returning 'Hello World'",
                "tools": [{
                    "name": "Write",
                    "description": "Write content to a file",
                    "input_schema": {
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    }
                }],
                "history": [
                    Message(role="user", content="Create a basic main.py file"),
                    Message(role="assistant", content="<tool_call>{\"tool\": \"Write\", \"input\": {\"file_path\": \"main.py\", \"content\": \"print('hello')\"}}</tool_call>"),
                    Message(role="tool", content="[Tool result]: File created successfully"),
                ]
            },
            {
                "name": "Complex request with context",
                "description": "Create requirements.txt with 'fastapi==0.115.0' and 'uvicorn[standard]==0.30.6'",
                "tools": [
                    {
                        "name": "Write",
                        "description": "Write content to a file",
                        "input_schema": {
                            "properties": {
                                "file_path": {"type": "string"},
                                "content": {"type": "string"}
                            }
                        }
                    },
                    {
                        "name": "Bash",
                        "description": "Execute a bash command",
                        "input_schema": {
                            "properties": {
                                "command": {"type": "string"}
                            }
                        }
                    }
                ],
                "history": [
                    Message(role="user", content="Help me create a web application"),
                    Message(role="assistant", content="I'd be happy to help you create a web application. What type of framework would you like to use?"),
                    Message(role="user", content="Use FastAPI, it's a modern Python web framework"),
                ]
            },
            {
                "name": "Agent task with subtasks",
                "description": "Create a Python script at 'fetch_data.py' that fetches data from 'https://jsonplaceholder.typicode.com/todos/1' and saves it to 'data.json'",
                "tools": [
                    {
                        "name": "Write",
                        "description": "Write content to a file",
                        "input_schema": {
                            "properties": {
                                "file_path": {"type": "string"},
                                "content": {"type": "string"}
                            }
                        }
                    }
                ],
                "history": []
            }
        ]
        
        # Run each test
        for test in test_cases:
            await tester.run_test(
                test["name"],
                test["description"],
                test["tools"],
                test.get("history", [])
            )
            # Wait between tests to avoid rate limiting
            await asyncio.sleep(2)
        
        tester.print_summary()
        
    finally:
        await tester.teardown()


if __name__ == "__main__":
    asyncio.run(main())