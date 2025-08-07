"""
Main System Orchestrator for the Pokémon Deep Research Agent.

This module provides the main entry point and coordination layer for the entire
multi-agent system, managing agent lifecycle, user interactions, and system state.
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..agents.base_agent import BaseAgent
from ..agents.coordinator_agent import CoordinatorAgent
from ..agents.task_manager_agent import TaskManagerAgent
from ..agents.pokeapi_research_agent import PokéAPIResearchAgent
from ..agents.battle_strategy_agent import BattleStrategyAgent
from ..agents.pokemon_analysis_agent import PokemonAnalysisAgent
from ..agents.report_generation_agent import ReportGenerationAgent

from ..core.communication import MessageBus, TaskChannel
from ..core.vector_memory import VectorMemorySystem, create_vector_memory_system
from ..core.models import (
    AgentConfig, AgentType, AgentCapability, Task, TaskPriority, TaskStatus
)
from ..tools.pokeapi_client import PokéAPIClient
from ..tools.llm_client import LLMClient


class PokemonDeepResearchSystem:
    """
    Main orchestrator for the Pokemon Deep Research Agent system.
    
    This class manages the entire multi-agent system, including:
    - Agent lifecycle management
    - System initialization and shutdown
    - User interaction interface
    - System monitoring and health checks
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_id = uuid4()
        self.start_time: Optional[datetime] = None
        self.is_running = False
        
        # Core components
        self.message_bus: Optional[MessageBus] = None
        self.task_channel: Optional[TaskChannel] = None
        self.llm_client: Optional[LLMClient] = None
        self.pokeapi_client: Optional[PokéAPIClient] = None
        self.vector_memory: Optional[VectorMemorySystem] = None
        
        # Agents
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        
        # System state
        self.active_queries: Dict[UUID, Dict[str, Any]] = {}
        self.system_stats = {
            "queries_processed": 0,
            "total_tasks_completed": 0,
            "total_errors": 0,
            "uptime_seconds": 0
        }
        
        # Logging
        self.logger = self._setup_logging()
        
        # Shutdown handling
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup system logging."""
        log_level = self.config.get("logging", {}).get("level", "INFO")
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("pokemon_research_system.log")
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
    
    async def initialize(self) -> None:
        """Initialize the entire system."""
        self.logger.info("Initializing Pokemon Deep Research System...")
        
        try:
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize agents
            await self._initialize_agents()
            
            # Start all agents
            await self._start_agents()
            
            # System health check
            await self._system_health_check()
            
            self.is_running = True
            self.start_time = datetime.now()
            
            self.logger.info("Pokemon Deep Research System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            await self.shutdown()
            raise
    
    async def _initialize_core_components(self) -> None:
        """Initialize core system components."""
        
        # Initialize message bus
        self.message_bus = MessageBus()
        await self.message_bus.start()
        
        # Initialize task channel
        self.task_channel = TaskChannel()
        await self.task_channel.start()
        
        # Initialize LLM client
        openai_api_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.llm_client = LLMClient(
            api_key=openai_api_key,
            model=self.config.get("llm_model", "gpt-4.1-mini"),
            max_tokens=self.config.get("llm_max_tokens", 4000),
            temperature=self.config.get("llm_temperature", 0.7),
            cache_responses=self.config.get("llm_cache", True)
        )
        
        # Initialize PokéAPI client
        self.pokeapi_client = PokéAPIClient(
            base_url=self.config.get("pokeapi_base_url", "https://pokeapi.co/api/v2/"),
            cache_dir=self.config.get("cache_dir", "data/cache"),
            rate_limit_delay=self.config.get("pokeapi_rate_limit", 0.1),
            max_concurrent_requests=self.config.get("pokeapi_max_concurrent", 10)
        )
        
        # Initialize Vector Memory System
        self.vector_memory = await create_vector_memory_system(
            openai_client=self.llm_client.client,
            config={
                "memory_dir": self.config.get("memory_dir", "data/memory"),
                "embedding_model": self.config.get("embedding_model", "text-embedding-3-small"),
                "max_memory_entries": self.config.get("max_memory_entries", 10000),
                "similarity_threshold": self.config.get("similarity_threshold", 0.7)
            }
        )
        
        self.logger.info("Core components initialized")
    
    async def _initialize_agents(self) -> None:
        """Initialize all system agents."""
        
        # Define agent configurations
        self.agent_configs = {
            "coordinator": AgentConfig(
                agent_id="coordinator",
                agent_type=AgentType.COORDINATOR,
                name="Coordinator Agent",
                description="Main coordination and workflow management",
                capabilities=[
                    AgentCapability(name="query_processing", description="Process user queries"),
                    AgentCapability(name="workflow_coordination", description="Coordinate agent workflows"),
                    AgentCapability(name="result_synthesis", description="Synthesize final results")
                ],
                max_concurrent_tasks=5
            ),
            
            "task_manager": AgentConfig(
                agent_id="task_manager",
                agent_type=AgentType.TASK_MANAGER,
                name="Task Manager Agent",
                description="Query decomposition and task management",
                capabilities=[
                    AgentCapability(name="query_decomposition", description="Break down complex queries"),
                    AgentCapability(name="task_optimization", description="Optimize task execution"),
                    AgentCapability(name="dependency_management", description="Manage task dependencies")
                ],
                max_concurrent_tasks=3
            ),
            
            "pokeapi_researcher": AgentConfig(
                agent_id="pokeapi_researcher",
                agent_type=AgentType.POKEAPI_RESEARCHER,
                name="PokéAPI Research Agent",
                description="Pokemon data retrieval and preprocessing",
                capabilities=[
                    AgentCapability(name="pokemon_data_retrieval", description="Retrieve Pokemon data"),
                    AgentCapability(name="move_data_retrieval", description="Retrieve move data"),
                    AgentCapability(name="type_data_retrieval", description="Retrieve type data"),
                    AgentCapability(name="ability_data_retrieval", description="Retrieve ability data"),
                    AgentCapability(name="location_data_retrieval", description="Retrieve location data")
                ],
                max_concurrent_tasks=10
            ),
            
            "battle_strategist": AgentConfig(
                agent_id="battle_strategist",
                agent_type=AgentType.BATTLE_STRATEGIST,
                name="Battle Strategy Agent",
                description="Competitive analysis and strategy development",
                capabilities=[
                    AgentCapability(name="team_analysis", description="Analyze team compositions"),
                    AgentCapability(name="competitive_analysis", description="Competitive viability analysis"),
                    AgentCapability(name="matchup_analysis", description="Pokemon matchup analysis"),
                    AgentCapability(name="strategy_development", description="Develop battle strategies")
                ],
                max_concurrent_tasks=5
            ),
            
            "pokemon_analyst": AgentConfig(
                agent_id="pokemon_analyst",
                agent_type=AgentType.POKEMON_ANALYST,
                name="Pokemon Analysis Agent",
                description="Detailed Pokemon statistical analysis",
                capabilities=[
                    AgentCapability(name="stat_analysis", description="Statistical analysis"),
                    AgentCapability(name="ability_analysis", description="Ability analysis"),
                    AgentCapability(name="move_analysis", description="Move analysis"),
                    AgentCapability(name="type_effectiveness_analysis", description="Type effectiveness analysis")
                ],
                max_concurrent_tasks=8
            ),
            
            "report_generator": AgentConfig(
                agent_id="report_generator",
                agent_type=AgentType.REPORT_GENERATOR,
                name="Report Generation Agent",
                description="Research synthesis and report generation",
                capabilities=[
                    AgentCapability(name="report_synthesis", description="Synthesize research reports"),
                    AgentCapability(name="data_visualization", description="Create data visualizations"),
                    AgentCapability(name="recommendation_generation", description="Generate recommendations")
                ],
                max_concurrent_tasks=3
            )
        }
        
        # Create agent instances
        self.agents["coordinator"] = CoordinatorAgent(
            self.agent_configs["coordinator"],
            self.message_bus,
            self.task_channel,
            self.llm_client
        )
        
        self.agents["task_manager"] = TaskManagerAgent(
            self.agent_configs["task_manager"],
            self.message_bus,
            self.task_channel,
            self.llm_client
        )
        
        self.agents["pokeapi_researcher"] = PokéAPIResearchAgent(
            self.agent_configs["pokeapi_researcher"],
            self.message_bus,
            self.task_channel,
            self.pokeapi_client
        )
        
        self.agents["battle_strategist"] = BattleStrategyAgent(
            self.agent_configs["battle_strategist"],
            self.message_bus,
            self.task_channel,
            self.llm_client
        )
        
        self.agents["pokemon_analyst"] = PokemonAnalysisAgent(
            self.agent_configs["pokemon_analyst"],
            self.message_bus,
            self.task_channel,
            self.llm_client
        )
        
        self.agents["report_generator"] = ReportGenerationAgent(
            self.agent_configs["report_generator"],
            self.message_bus,
            self.task_channel,
            self.llm_client
        )
        
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    async def _start_agents(self) -> None:
        """Start all agents."""
        start_tasks = []
        
        for agent_id, agent in self.agents.items():
            self.logger.info(f"Starting agent: {agent_id}")
            start_tasks.append(agent.start())
        
        await asyncio.gather(*start_tasks)
        self.logger.info("All agents started successfully")
    
    async def _system_health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            "system_id": str(self.system_id),
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "agents": {}
        }
        
        # Check core components
        health_status["components"]["message_bus"] = "healthy" if self.message_bus else "unhealthy"
        health_status["components"]["task_channel"] = "healthy" if self.task_channel else "unhealthy"
        health_status["components"]["llm_client"] = "healthy" if self.llm_client else "unhealthy"
        health_status["components"]["pokeapi_client"] = "healthy" if self.pokeapi_client else "unhealthy"
        
        # Check agents
        for agent_id, agent in self.agents.items():
            agent_status = agent.get_status()
            health_status["agents"][agent_id] = {
                "status": "healthy" if agent.is_running else "unhealthy",
                "current_tasks": agent_status.get("current_tasks", 0),
                "completed_tasks": agent_status.get("completed_tasks", 0)
            }
        
        # Determine overall status
        unhealthy_components = [k for k, v in health_status["components"].items() if v == "unhealthy"]
        unhealthy_agents = [k for k, v in health_status["agents"].items() if v["status"] == "unhealthy"]
        
        if unhealthy_components or unhealthy_agents:
            health_status["overall_status"] = "degraded"
            if len(unhealthy_components) > 1 or len(unhealthy_agents) > 2:
                health_status["overall_status"] = "unhealthy"
        
        self.logger.info(f"System health check: {health_status['overall_status']}")
        return health_status
    
    async def process_user_query(self, query: str, user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system.
        
        Args:
            query: The user's research query
            user_preferences: Optional user preferences and constraints
            
        Returns:
            Dict containing the research results and metadata
        """
        if not self.is_running:
            raise RuntimeError("System is not running")
        
        query_id = uuid4()
        self.logger.info(f"Processing user query {query_id}: {query}")
        
        try:
            # Create research query task
            research_task = Task(
                title=f"Research Query: {query[:50]}...",
                description=f"Process user research query: {query}",
                task_type="research_query",
                priority=TaskPriority.HIGH,
                metadata={
                    "query_data": {
                        "query": query,
                        "parameters": user_preferences or {},
                        "preferences": user_preferences or {}
                    },
                    "query_id": str(query_id)
                }
            )
            
            # Submit to coordinator
            await self.task_channel.submit_task(research_task)
            
            # Track query
            self.active_queries[query_id] = {
                "query": query,
                "task_id": research_task.id,
                "start_time": datetime.now(),
                "status": "processing",
                "preferences": user_preferences or {}
            }
            
            # Wait for completion with timeout
            timeout = self.config.get("query_timeout", 300)  # 5 minutes default
            result = await self._wait_for_query_completion(query_id, timeout)
            
            # Update stats
            self.system_stats["queries_processed"] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            self.system_stats["total_errors"] += 1
            
            return {
                "query_id": str(query_id),
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        finally:
            # Cleanup
            if query_id in self.active_queries:
                del self.active_queries[query_id]
    
    async def _wait_for_query_completion(self, query_id: UUID, timeout: int) -> Dict[str, Any]:
        """Wait for query completion with timeout."""
        
        start_time = datetime.now()
        check_interval = 2.0  # Check every 2 seconds
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            # Check if shutdown requested
            if self._shutdown_requested:
                return {
                    "query_id": str(query_id),
                    "status": "cancelled",
                    "message": "System shutdown requested",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Check task status
            query_info = self.active_queries.get(query_id)
            if not query_info:
                break
            
            task_id = query_info["task_id"]
            task_status = await self.task_channel.get_task_status(task_id)
            
            if task_status == TaskStatus.COMPLETED:
                # Get result
                result = await self.task_channel.get_task_result(task_id)
                
                return {
                    "query_id": str(query_id),
                    "status": "completed",
                    "result": result,
                    "processing_time": (datetime.now() - query_info["start_time"]).total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
            
            elif task_status == TaskStatus.FAILED:
                # Get error information
                error_info = await self.task_channel.get_task_error(task_id)
                
                return {
                    "query_id": str(query_id),
                    "status": "failed",
                    "error": error_info,
                    "processing_time": (datetime.now() - query_info["start_time"]).total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Wait before next check
            await asyncio.sleep(check_interval)
        
        # Timeout reached
        return {
            "query_id": str(query_id),
            "status": "timeout",
            "message": f"Query processing timed out after {timeout} seconds",
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        if not self.is_running:
            return {
                "status": "stopped",
                "message": "System is not running"
            }
        
        # Calculate uptime
        uptime_seconds = 0
        if self.start_time:
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        # Get agent statuses
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = agent.get_status()
        
        # Get component statuses
        component_status = {
            "message_bus": {
                "status": "healthy" if self.message_bus else "unhealthy",
                "active_subscriptions": len(self.message_bus.subscriptions) if self.message_bus else 0
            },
            "task_channel": {
                "status": "healthy" if self.task_channel else "unhealthy",
                "pending_tasks": len(self.task_channel.pending_tasks) if self.task_channel else 0
            },
            "llm_client": {
                "status": "healthy" if self.llm_client else "unhealthy",
                "usage_stats": self.llm_client.get_usage_stats() if self.llm_client else {}
            },
            "pokeapi_client": {
                "status": "healthy" if self.pokeapi_client else "unhealthy",
                "cache_stats": self.pokeapi_client.get_cache_stats() if self.pokeapi_client else {}
            }
        }
        
        return {
            "system_id": str(self.system_id),
            "status": "running",
            "uptime_seconds": uptime_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "active_queries": len(self.active_queries),
            "system_stats": self.system_stats,
            "agents": agent_statuses,
            "components": component_status,
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system."""
        if not self.is_running:
            return
        
        self.logger.info("Initiating system shutdown...")
        self.is_running = False
        
        try:
            # Stop all agents
            if self.agents:
                self.logger.info("Stopping agents...")
                stop_tasks = []
                for agent_id, agent in self.agents.items():
                    self.logger.info(f"Stopping agent: {agent_id}")
                    stop_tasks.append(agent.stop())
                
                await asyncio.gather(*stop_tasks, return_exceptions=True)
            
            # Shutdown core components
            if self.pokeapi_client:
                await self.pokeapi_client.close()
            
            if self.task_channel:
                await self.task_channel.stop()
            
            if self.message_bus:
                await self.message_bus.stop()
            
            # Clear LLM client cache
            if self.llm_client:
                self.llm_client.clear_cache()
            
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def run_interactive_mode(self) -> None:
        """Run the system in interactive mode."""
        
        print("🔍 Pokemon Deep Research Agent System")
        print("=" * 50)
        print("Welcome to the Pokemon Deep Research Agent!")
        print("Ask me anything about Pokemon and I'll provide comprehensive research.")
        print("Type 'help' for commands, 'quit' to exit.")
        print()
        
        while self.is_running and not self._shutdown_requested:
            try:
                # Get user input
                user_input = input("🎯 Your Pokemon question: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._print_help()
                    continue
                
                elif user_input.lower() == 'status':
                    await self._print_system_status()
                    continue
                
                elif user_input.lower().startswith('clear'):
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                # Process query
                print("🔄 Processing your query...")
                print()
                
                result = await self.process_user_query(user_input)
                
                # Display result
                await self._display_query_result(result)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            
            except Exception as e:
                print(f"❌ Error: {e}")
                self.logger.error(f"Interactive mode error: {e}")
    
    def _print_help(self) -> None:
        """Print help information."""
        print()
        print("📚 Available Commands:")
        print("  help     - Show this help message")
        print("  status   - Show system status")
        print("  clear    - Clear the screen")
        print("  quit     - Exit the system")
        print()
        print("🎯 Example Queries:")
        print("  • What are the best Fire-type Pokemon for beginners?")
        print("  • Compare Charizard vs Blastoise for competitive play")
        print("  • Build me a team around Garchomp for singles")
        print("  • What Pokemon can I find in Ruby/Sapphire caves?")
        print("  • Analyze Pikachu's stats and competitive viability")
        print()
    
    async def _print_system_status(self) -> None:
        """Print system status."""
        status = await self.get_system_status()
        
        print()
        print("📊 System Status:")
        print(f"  Status: {status['status']}")
        print(f"  Uptime: {status['uptime_seconds']:.1f} seconds")
        print(f"  Active Queries: {status['active_queries']}")
        print(f"  Queries Processed: {status['system_stats']['queries_processed']}")
        print()
        
        print("🤖 Agents:")
        for agent_id, agent_status in status['agents'].items():
            running_status = "✅" if agent_status['is_running'] else "❌"
            print(f"  {running_status} {agent_status['name']}: {agent_status['completed_tasks']} tasks completed")
        print()
    
    async def _display_query_result(self, result: Dict[str, Any]) -> None:
        """Display query result to user."""
        
        print("📋 Research Results:")
        print("=" * 50)
        
        status = result.get("status", "unknown")
        
        if status == "completed":
            # Display successful result
            query_result = result.get("result", {})
            
            if "synthesis" in query_result:
                synthesis = query_result["synthesis"]
                print(f"📊 Analysis: {synthesis.get('synthesis_text', 'Analysis completed')}")
                print()
                
                # Display key insights
                insights = synthesis.get("key_insights", [])
                if insights:
                    print("💡 Key Insights:")
                    for i, insight in enumerate(insights[:5], 1):
                        print(f"  {i}. {insight.get('insight', 'Insight available')}")
                    print()
            
            elif "markdown_report" in query_result:
                # Display markdown report
                markdown = query_result["markdown_report"]
                print(markdown)
                print()
            
            else:
                # Display basic result info
                print(f"✅ Query completed successfully")
                print(f"📈 Confidence: {query_result.get('confidence', 'Unknown')}")
                print(f"📚 Sources: {', '.join(query_result.get('sources', ['Analysis']))}")
                print()
            
            # Display processing time
            processing_time = result.get("processing_time", 0)
            print(f"⏱️  Processing time: {processing_time:.1f} seconds")
            
        elif status == "error":
            print(f"❌ Error: {result.get('error', 'Unknown error occurred')}")
            
        elif status == "timeout":
            print(f"⏰ Timeout: {result.get('message', 'Query processing timed out')}")
            
        elif status == "failed":
            print(f"💥 Failed: {result.get('error', 'Query processing failed')}")
            
        else:
            print(f"❓ Unknown status: {status}")
        
        print("=" * 50)
        print()


async def main():
    """Main entry point for the system."""
    
    # Default configuration
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "llm_model": "gpt-4",
        "llm_max_tokens": 4000,
        "llm_temperature": 0.7,
        "llm_cache": True,
        "pokeapi_base_url": "https://pokeapi.co/api/v2/",
        "cache_dir": "data/cache",
        "pokeapi_rate_limit": 0.1,
        "pokeapi_max_concurrent": 10,
        "query_timeout": 300,
        "logging": {
            "level": "INFO"
        }
    }
    
    # Create and initialize system
    system = PokemonDeepResearchSystem(config)
    
    try:
        await system.initialize()
        await system.run_interactive_mode()
        
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        
    except Exception as e:
        print(f"System error: {e}")
        
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

