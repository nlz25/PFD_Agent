"""
Google AI Agent Demo
A demo agent using Google's Generative AI SDK with tool calling capabilities.
"""

import os
import asyncio
from typing import Any, Dict, List, Optional
from google.generativeai.agent import Agent


class GoogleADKAgent:
    """An agent using Google's ADK Agent API."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """Initialize the Google ADK Agent.

        Args:
            api_key: Google AI API key. If None, will use GOOGLE_API_KEY env var.
            model_name: Name of the model to use.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")

        self.model_name = model_name
        self.agent = Agent(api_key=self.api_key, model=self.model_name)
        self.conversation_history = []

    async def chat(self, message: str) -> str:
        """Send a message to the agent and get a response."""
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": message})

            # Generate response using the ADK Agent
            response = self.agent.chat(message)

            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history.copy()

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()


async def main():
    """Demo usage of the Google ADK Agent."""
    print("🤖 Google ADK Agent Demo")
    print("=" * 50)

    try:
        # Initialize the agent
        agent = GoogleADKAgent()
        print("✅ Agent initialized successfully!")

        # Demo conversations
        demo_messages = [
            "Hello! What can you help me with?",
            "Can you calculate 15 * 7 + 23?",
            "What's the weather like in Tokyo?",
        ]

        for message in demo_messages:
            print(f"\n👤 User: {message}")
            response = await agent.chat(message)
            print(f"🤖 Agent: {response}")

        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure to set your GOOGLE_API_KEY environment variable.")


if __name__ == "__main__":
    asyncio.run(main())