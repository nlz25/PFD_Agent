#!/usr/bin/env python3
"""
Database Agent Test Script
Tests the DatabaseAgent functionality with a sample database.
"""
import sys
sys.path.append("../")  # Ensure parent directory is in the path
import os
import asyncio
from typing import Optional
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Optional
from google.adk.runners import Runner
from agents.pfd_agent.database_agent import get_agent_async
from agents.pfd_agent.utils.llm_config import create_default_config,LLMConfig


async def test_database_agent(
    llm_config,
    server_path: Optional[str] = None, 
    db_path: Optional[str] = None
    
):
    """Test the database agent with sample queries."""
    print("🗄️  Database Agent Test")
    print("=" * 50)
    
    try:
        # Set up test database path
        #test_db_path = db_path or (Path(__file__).parent.parent.parent.parent / ".tests" / "database" / "database.db")
        session_service = InMemorySessionService()
    # Artifact service might not be needed for this example
        artifacts_service = InMemoryArtifactService()

        session = await session_service.create_session(
          state={}, app_name='ase_db_database_app', user_id='user_fs'
  )
        print(f"✅ Session created with ID: {session.id}")
        # Initialize the database agent
        print("✅ Initializing database agent...")
        print(llm_config.model_type)
        agent, toolset = await get_agent_async(
            llm_config=llm_config,
            server_path=server_path,
            db_path=db_path
        )
        
        print(f"Agent name: {agent.name}")
        print(f"Server path: {agent.server_path}")
        print(f"Database path: {agent.db_path or 'Using default/env'}")
        
        # start running the service
        runner = Runner(
            app_name='ase_db_database_app',
            agent=agent,
            artifact_service=artifacts_service, # Optional
            session_service=session_service,
            )

        print("Running agent...")

        # Test queries
        test_queries = [
            "Can you query the database for 'Si32' compounds? Use a limit of 5.",
            "Can you export from the database at least 5 structures with 'Si32' compounds? And tell me where the files are saved.",
            #"What structures do we have for H2O in the database?",
            #"Show me the first 3 entries in the database and export them to ./test_export directory."
        ]
        
        print("\n" + "=" * 50)
        print("🔍 Testing Database Queries")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Test Query {i}:")
            content = types.Content(role='user', parts=[types.Part(text=query)])
            events_async = runner.run_async(
                session_id=session.id, user_id=session.user_id, new_message=content
            )
            
            async for event in events_async:
                print(f"Event received: {event}")
        
        print("\n" + "=" * 50)
        print("✅ Database agent initialized successfully!")
        print("💡 The agent is ready to handle database queries through MCP server.")
        
        await toolset.close()
        # Clean up
        if hasattr(agent, 'close'):
            await agent.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure the database server and test database are available.")


def main():
    """Main entry point."""
    LLMConfig.reset()
    llm_config = LLMConfig(
        model_type="dashscope/qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    )
    print(llm_config.custom_model)
    asyncio.run(test_database_agent(
        llm_config,
        server_path='/home/ruoyu/dev/PFD-Agent/server/database_server/server.py',
        db_path='/home/ruoyu/dev/PFD-Agent/server/database_server/database.db'
    ))

if __name__ == "__main__":
    main()