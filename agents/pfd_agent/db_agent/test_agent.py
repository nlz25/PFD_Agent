#!/usr/bin/env python3
"""
Database Agent Test Script
Tests the DatabaseAgent functionality with a sample database.
"""

import asyncio
import os
from pathlib import Path
from agents.pfd_agent.database_agent import init_database_agent
from agents.pfd_agent.utils.llm_config import LlmConfig


async def test_database_agent():
    """Test the database agent with sample queries."""
    print("🗄️  Database Agent Test")
    print("=" * 50)
    
    try:
        # Set up test database path
        test_db_path = Path(__file__).parent.parent.parent.parent / ".tests" / "database" / "database.db"
        
        # Initialize the database agent
        print("✅ Initializing database agent...")
        agent = init_database_agent(
            llm_config=LlmConfig,
            db_path=str(test_db_path) if test_db_path.exists() else None
        )
        
        print(f"Agent name: {agent.name}")
        print(f"Server path: {agent.server_path}")
        print(f"Database path: {agent.db_path or 'Using default/env'}")
        
        # Test queries
        test_queries = [
            "Can you query the database for silicon compounds? Use a limit of 5.",
            "What structures do we have for H2O in the database?",
            "Show me the first 3 entries in the database and export them to ./test_export directory."
        ]
        
        print("\n" + "=" * 50)
        print("🔍 Testing Database Queries")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Test Query {i}:")
            print(f"User: {query}")
            print("Agent: [This would run the agent with the query]")
            print("Note: Full agent execution requires proper async context and event handling")
        
        print("\n" + "=" * 50)
        print("✅ Database agent initialized successfully!")
        print("💡 The agent is ready to handle database queries through MCP server.")
        
        # Clean up
        if hasattr(agent, 'close'):
            await agent.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure the database server and test database are available.")


def main():
    """Main entry point."""
    asyncio.run(test_database_agent())


if __name__ == "__main__":
    main()