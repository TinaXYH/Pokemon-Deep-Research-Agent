#!/usr/bin/env python3
"""
Simple test script to verify LLM client functionality.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tools.llm_client import LLMClient


async def test_llm_client():
    """Test the LLM client with a simple query."""

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OPENAI_API_KEY found in environment")
        return False

    print(f"🔑 Using API key: {api_key[:20]}...")

    # Create LLM client
    client = LLMClient(
        api_key=api_key, model="gpt-4.1-mini", max_tokens=100, temperature=0.7
    )

    print(f"🤖 Created LLM client with model: {client.model}")

    # Test simple completion
    messages = [
        {"role": "user", "content": "What type is Pikachu? Answer in one sentence."}
    ]

    try:
        print("🔍 Testing chat completion...")
        response = await client.chat_completion(messages)

        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
            print(f"✅ LLM Response: {content}")
            return True
        else:
            print(f"❌ Unexpected response format: {response}")
            return False

    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return False


async def main():
    """Main test function."""
    print("🧪 Testing LLM Client")
    print("=" * 50)

    success = await test_llm_client()

    if success:
        print("\n🎉 LLM client test passed!")
    else:
        print("\n💥 LLM client test failed!")

    return success


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    # Run test
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
