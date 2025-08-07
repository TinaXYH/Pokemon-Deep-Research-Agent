"""
Task Manager Agent for the Pokémon Deep Research Agent system.

The Task Manager Agent acts as the "strategy lead" responsible for:
- Decomposing complex queries into actionable subtasks
- Defining task dependencies and execution order
- Optimizing task distribution for parallel execution
- Monitoring subtask completion and integration
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from ..agents.base_agent import BaseAgent
from ..core.communication import MessageBus, TaskChannel
from ..core.models import (
    AgentConfig, Message, MessageType, Task, TaskPriority
)
from ..tools.llm_client import LLMClient


class TaskManagerAgent(BaseAgent):
    """
    Task Manager Agent that decomposes queries into executable subtasks.
    
    This agent analyzes research queries and breaks them down into specific,
    actionable tasks that can be distributed to specialized agents.
    """
    
    def __init__(
        self, 
        config: AgentConfig, 
        message_bus: MessageBus, 
        task_channel: TaskChannel,
        llm_client: LLMClient
    ):
        super().__init__(config, message_bus, task_channel)
        self.llm_client = llm_client
        
        # Task decomposition templates
        self.decomposition_templates = {
            "team_building": self._decompose_team_building_query,
            "individual_analysis": self._decompose_individual_analysis_query,
            "comparison": self._decompose_comparison_query,
            "battle_strategy": self._decompose_battle_strategy_query,
            "location_info": self._decompose_location_query,
            "general": self._decompose_general_query
        }
        
        # Agent capability mapping
        self.agent_capabilities = {
            "pokeapi_researcher": [
                "pokemon_data_retrieval",
                "move_data_retrieval", 
                "type_data_retrieval",
                "ability_data_retrieval",
                "location_data_retrieval"
            ],
            "battle_strategist": [
                "team_analysis",
                "competitive_analysis",
                "matchup_analysis",
                "strategy_development"
            ],
            "pokemon_analyst": [
                "stat_analysis",
                "ability_analysis",
                "move_analysis",
                "type_effectiveness_analysis"
            ],
            "report_generator": [
                "report_synthesis",
                "data_visualization",
                "recommendation_generation"
            ]
        }
    
    async def _initialize(self) -> None:
        """Initialize task manager specific resources."""
        self.logger.info("Task Manager Agent initialized")
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process task manager specific tasks."""
        task_type = task.task_type
        
        if task_type == "decompose_query":
            return await self._handle_query_decomposition(task)
        elif task_type == "optimize_workflow":
            return await self._handle_workflow_optimization(task)
        elif task_type == "dependency_analysis":
            return await self._handle_dependency_analysis(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _handle_query_decomposition(self, task: Task) -> Dict[str, Any]:
        """Handle query decomposition requests."""
        query_data = task.metadata.get("query_data", {})
        query = query_data.get("query", "")
        query_type = query_data.get("query_type", "general")
        parameters = query_data.get("parameters", {})
        
        if not query:
            raise ValueError("No query provided for decomposition")
        
        # Get appropriate decomposition function
        decompose_func = self.decomposition_templates.get(
            query_type, 
            self._decompose_general_query
        )
        
        # Decompose the query
        subtasks = await decompose_func(query, parameters)
        
        # Optimize task order and dependencies
        optimized_tasks = await self._optimize_task_execution(subtasks)
        
        self.logger.info(f"Decomposed query into {len(optimized_tasks)} subtasks")
        
        return {
            "status": "decomposition_completed",
            "tasks": optimized_tasks,
            "original_query": query,
            "query_type": query_type
        }
    
    async def _decompose_team_building_query(
        self, 
        query: str, 
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose team building queries."""
        
        # Extract constraints from query and parameters
        type_constraint = parameters.get("type_constraint")
        game_version = parameters.get("game_version", "general")
        battle_format = parameters.get("battle_format", "singles")
        difficulty = parameters.get("difficulty_level", "intermediate")
        
        tasks = []
        
        # Task 1: Research available Pokemon
        if type_constraint:
            tasks.append({
                "title": f"Research {type_constraint} type Pokemon",
                "description": f"Retrieve comprehensive data for all {type_constraint} type Pokemon",
                "task_type": "pokemon_research",
                "required_capability": "pokemon_data_retrieval",
                "priority": 3,
                "complexity": "medium",
                "parameters": {
                    "type_filter": type_constraint,
                    "game_version": game_version,
                    "include_stats": True,
                    "include_moves": True,
                    "include_abilities": True
                }
            })
        else:
            tasks.append({
                "title": "Research competitive Pokemon pool",
                "description": "Retrieve data for commonly used competitive Pokemon",
                "task_type": "pokemon_research",
                "required_capability": "pokemon_data_retrieval",
                "priority": 3,
                "complexity": "high",
                "parameters": {
                    "competitive_filter": True,
                    "game_version": game_version,
                    "battle_format": battle_format,
                    "tier_filter": "OU,UU,RU" if difficulty == "competitive" else "all"
                }
            })
        
        # Task 2: Analyze team synergies and roles
        tasks.append({
            "title": "Analyze team composition requirements",
            "description": "Determine optimal team roles and synergies",
            "task_type": "team_analysis",
            "required_capability": "team_analysis",
            "priority": 2,
            "complexity": "high",
            "parameters": {
                "battle_format": battle_format,
                "type_constraint": type_constraint,
                "difficulty_level": difficulty
            },
            "dependencies": [0]  # Depends on Pokemon research
        })
        
        # Task 3: Evaluate competitive viability
        tasks.append({
            "title": "Evaluate competitive viability",
            "description": "Assess Pokemon viability in current meta",
            "task_type": "competitive_analysis",
            "required_capability": "competitive_analysis",
            "priority": 2,
            "complexity": "medium",
            "parameters": {
                "battle_format": battle_format,
                "meta_analysis": True,
                "usage_stats": True
            },
            "dependencies": [0]  # Depends on Pokemon research
        })
        
        # Task 4: Generate team recommendations
        tasks.append({
            "title": "Generate team recommendations",
            "description": "Create specific team recommendations with movesets",
            "task_type": "team_recommendation",
            "required_capability": "strategy_development",
            "priority": 1,
            "complexity": "high",
            "parameters": {
                "team_size": 6,
                "include_movesets": True,
                "include_items": True,
                "include_evs": True if difficulty == "competitive" else False
            },
            "dependencies": [1, 2]  # Depends on analysis tasks
        })
        
        return tasks
    
    async def _decompose_individual_analysis_query(
        self, 
        query: str, 
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose individual Pokemon analysis queries."""
        
        pokemon_mentioned = parameters.get("pokemon_mentioned", [])
        game_version = parameters.get("game_version", "general")
        analysis_depth = parameters.get("difficulty_level", "intermediate")
        
        tasks = []
        
        # If specific Pokemon mentioned, analyze them
        if pokemon_mentioned:
            for pokemon in pokemon_mentioned:
                tasks.append({
                    "title": f"Analyze {pokemon}",
                    "description": f"Comprehensive analysis of {pokemon}",
                    "task_type": "pokemon_analysis",
                    "required_capability": "stat_analysis",
                    "priority": 3,
                    "complexity": "medium",
                    "parameters": {
                        "pokemon_name": pokemon,
                        "game_version": game_version,
                        "include_movesets": True,
                        "include_abilities": True,
                        "include_stats": True,
                        "competitive_analysis": analysis_depth == "competitive"
                    }
                })
        else:
            # General analysis task - need to identify Pokemon first
            tasks.append({
                "title": "Identify target Pokemon",
                "description": "Identify Pokemon matching query criteria",
                "task_type": "pokemon_identification",
                "required_capability": "pokemon_data_retrieval",
                "priority": 3,
                "complexity": "low",
                "parameters": {
                    "query_context": query,
                    "game_version": game_version
                }
            })
            
            tasks.append({
                "title": "Analyze identified Pokemon",
                "description": "Detailed analysis of identified Pokemon",
                "task_type": "pokemon_analysis",
                "required_capability": "stat_analysis",
                "priority": 2,
                "complexity": "medium",
                "parameters": {
                    "analysis_depth": analysis_depth,
                    "game_version": game_version
                },
                "dependencies": [0]
            })
        
        # Add usage and training analysis if relevant
        if "easy" in query.lower() or "train" in query.lower():
            tasks.append({
                "title": "Analyze training requirements",
                "description": "Evaluate ease of training and usage",
                "task_type": "training_analysis",
                "required_capability": "pokemon_analyst",
                "priority": 2,
                "complexity": "low",
                "parameters": {
                    "focus": "training_ease",
                    "beginner_friendly": True
                },
                "dependencies": [len(tasks) - 1] if tasks else []
            })
        
        return tasks
    
    async def _decompose_comparison_query(
        self, 
        query: str, 
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose Pokemon comparison queries."""
        
        pokemon_mentioned = parameters.get("pokemon_mentioned", [])
        comparison_criteria = parameters.get("comparison_criteria", ["stats", "abilities"])
        
        tasks = []
        
        # Research each Pokemon being compared
        for pokemon in pokemon_mentioned:
            tasks.append({
                "title": f"Research {pokemon} data",
                "description": f"Gather comprehensive data for {pokemon}",
                "task_type": "pokemon_research",
                "required_capability": "pokemon_data_retrieval",
                "priority": 3,
                "complexity": "medium",
                "parameters": {
                    "pokemon_name": pokemon,
                    "detailed_analysis": True
                }
            })
        
        # Comparative analysis task
        tasks.append({
            "title": "Perform comparative analysis",
            "description": "Compare Pokemon across specified criteria",
            "task_type": "comparative_analysis",
            "required_capability": "stat_analysis",
            "priority": 2,
            "complexity": "high",
            "parameters": {
                "pokemon_list": pokemon_mentioned,
                "comparison_criteria": comparison_criteria,
                "include_recommendations": True
            },
            "dependencies": list(range(len(pokemon_mentioned)))
        })
        
        return tasks
    
    async def _decompose_battle_strategy_query(
        self, 
        query: str, 
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose battle strategy queries."""
        
        battle_format = parameters.get("battle_format", "singles")
        pokemon_mentioned = parameters.get("pokemon_mentioned", [])
        
        tasks = []
        
        # Research Pokemon involved
        if pokemon_mentioned:
            for pokemon in pokemon_mentioned:
                tasks.append({
                    "title": f"Research {pokemon} battle data",
                    "description": f"Gather battle-relevant data for {pokemon}",
                    "task_type": "battle_research",
                    "required_capability": "pokemon_data_retrieval",
                    "priority": 3,
                    "complexity": "medium",
                    "parameters": {
                        "pokemon_name": pokemon,
                        "battle_format": battle_format,
                        "include_movesets": True,
                        "include_counters": True
                    }
                })
        
        # Strategy development
        tasks.append({
            "title": "Develop battle strategy",
            "description": "Create comprehensive battle strategy",
            "task_type": "strategy_development",
            "required_capability": "strategy_development",
            "priority": 2,
            "complexity": "high",
            "parameters": {
                "battle_format": battle_format,
                "pokemon_focus": pokemon_mentioned,
                "include_counters": True,
                "include_synergies": True
            },
            "dependencies": list(range(len(pokemon_mentioned))) if pokemon_mentioned else []
        })
        
        return tasks
    
    async def _decompose_location_query(
        self, 
        query: str, 
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose location-based queries."""
        
        game_version = parameters.get("game_version", "general")
        location_mentioned = parameters.get("location_mentioned", [])
        
        tasks = []
        
        # Location research
        if location_mentioned:
            for location in location_mentioned:
                tasks.append({
                    "title": f"Research {location} Pokemon",
                    "description": f"Find Pokemon available at {location}",
                    "task_type": "location_research",
                    "required_capability": "location_data_retrieval",
                    "priority": 3,
                    "complexity": "medium",
                    "parameters": {
                        "location_name": location,
                        "game_version": game_version,
                        "include_encounter_rates": True
                    }
                })
        else:
            # General location search based on criteria
            tasks.append({
                "title": "Search Pokemon locations",
                "description": "Find Pokemon matching location criteria",
                "task_type": "location_search",
                "required_capability": "location_data_retrieval",
                "priority": 3,
                "complexity": "medium",
                "parameters": {
                    "search_criteria": query,
                    "game_version": game_version
                }
            })
        
        return tasks
    
    async def _decompose_general_query(
        self, 
        query: str, 
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose general queries that don't fit specific patterns."""
        
        # Use LLM to help with decomposition
        try:
            decomposition_prompt = f"""
            Break down this Pokemon research query into 2-4 specific, actionable subtasks:
            
            Query: {query}
            Parameters: {parameters}
            
            Each subtask should:
            1. Have a clear, specific objective
            2. Be assignable to a specialized agent
            3. Produce concrete, measurable results
            4. Build toward answering the original query
            
            Available agent capabilities:
            - pokemon_data_retrieval: Get Pokemon stats, moves, abilities
            - competitive_analysis: Analyze competitive viability and usage
            - stat_analysis: Detailed statistical analysis and comparisons
            - strategy_development: Create battle strategies and recommendations
            
            Return a JSON array of task objects with: title, description, required_capability, priority (1-3), complexity (low/medium/high)
            """
            
            response = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a Pokemon research task planner."},
                    {"role": "user", "content": decomposition_prompt}
                ],
                temperature=0.3
            )        
            import json
            result = json.loads(response["choices"][0]["message"]["content"])
            tasks = result.get("tasks", [])
            
            # Add default fields
            for i, task in enumerate(tasks):
                task.setdefault("task_type", "general_research")
                task.setdefault("parameters", {"query": query})
                task.setdefault("priority", 2)
                task.setdefault("complexity", "medium")
            
            return tasks
            
        except Exception as e:
            self.logger.error(f"LLM decomposition failed: {e}")
            
            # Fallback to basic decomposition
            return [{
                "title": "General Pokemon research",
                "description": f"Research query: {query}",
                "task_type": "general_research",
                "required_capability": "pokemon_data_retrieval",
                "priority": 2,
                "complexity": "medium",
                "parameters": {"query": query, "parameters": parameters}
            }]
    
    async def _optimize_task_execution(
        self, 
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Optimize task execution order and dependencies."""
        
        # Add task IDs for dependency tracking
        for i, task in enumerate(tasks):
            task["task_id"] = i
            task.setdefault("dependencies", [])
        
        # Sort by priority (higher priority first) and complexity
        def task_sort_key(task):
            priority = task.get("priority", 2)
            complexity_weight = {"low": 1, "medium": 2, "high": 3}
            complexity = complexity_weight.get(task.get("complexity", "medium"), 2)
            return (-priority, complexity)  # Negative priority for descending order
        
        # Separate tasks with and without dependencies
        independent_tasks = [t for t in tasks if not t.get("dependencies")]
        dependent_tasks = [t for t in tasks if t.get("dependencies")]
        
        # Sort independent tasks by priority
        independent_tasks.sort(key=task_sort_key)
        
        # Sort dependent tasks by their dependencies and priority
        dependent_tasks.sort(key=lambda t: (min(t["dependencies"]), -t.get("priority", 2)))
        
        # Combine optimized task list
        optimized_tasks = independent_tasks + dependent_tasks
        
        # Add execution estimates
        for task in optimized_tasks:
            task["estimated_duration"] = self._estimate_task_duration(task)
        
        return optimized_tasks
    
    def _estimate_task_duration(self, task: Dict[str, Any]) -> int:
        """Estimate task duration in seconds."""
        base_duration = {
            "low": 15,
            "medium": 45,
            "high": 120
        }
        
        complexity = task.get("complexity", "medium")
        duration = base_duration.get(complexity, 45)
        
        # Adjust based on task type
        task_type = task.get("task_type", "")
        if "research" in task_type:
            duration += 30
        elif "analysis" in task_type:
            duration += 60
        elif "recommendation" in task_type:
            duration += 90
        
        return duration
    
    async def _handle_workflow_optimization(self, task: Task) -> Dict[str, Any]:
        """Handle workflow optimization requests."""
        workflow_data = task.metadata.get("workflow_data", {})
        current_tasks = workflow_data.get("tasks", [])
        
        # Analyze current workflow for bottlenecks
        bottlenecks = self._identify_bottlenecks(current_tasks)
        
        # Suggest optimizations
        optimizations = self._suggest_optimizations(current_tasks, bottlenecks)
        
        return {
            "status": "optimization_completed",
            "bottlenecks": bottlenecks,
            "optimizations": optimizations,
            "estimated_improvement": self._calculate_improvement(optimizations)
        }
    
    def _identify_bottlenecks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential bottlenecks in task execution."""
        bottlenecks = []
        
        # Check for tasks with many dependencies
        for task in tasks:
            deps = task.get("dependencies", [])
            if len(deps) > 2:
                bottlenecks.append({
                    "type": "dependency_bottleneck",
                    "task": task["title"],
                    "issue": f"Task depends on {len(deps)} other tasks"
                })
        
        # Check for high-complexity tasks without parallelization
        high_complexity_tasks = [t for t in tasks if t.get("complexity") == "high"]
        if len(high_complexity_tasks) > 2:
            bottlenecks.append({
                "type": "complexity_bottleneck",
                "issue": f"{len(high_complexity_tasks)} high-complexity tasks may cause delays"
            })
        
        return bottlenecks
    
    def _suggest_optimizations(
        self, 
        tasks: List[Dict[str, Any]], 
        bottlenecks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest workflow optimizations."""
        optimizations = []
        
        # Suggest task parallelization
        parallelizable_tasks = [t for t in tasks if not t.get("dependencies")]
        if len(parallelizable_tasks) > 1:
            optimizations.append({
                "type": "parallelization",
                "description": f"Execute {len(parallelizable_tasks)} independent tasks in parallel",
                "estimated_time_savings": "30-50%"
            })
        
        # Suggest dependency reduction
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "dependency_bottleneck":
                optimizations.append({
                    "type": "dependency_reduction",
                    "description": "Consider breaking down complex dependent tasks",
                    "estimated_time_savings": "20-30%"
                })
        
        return optimizations
    
    def _calculate_improvement(self, optimizations: List[Dict[str, Any]]) -> str:
        """Calculate estimated improvement from optimizations."""
        if not optimizations:
            return "0%"
        
        # Simple heuristic based on optimization types
        total_improvement = 0
        for opt in optimizations:
            if opt["type"] == "parallelization":
                total_improvement += 40
            elif opt["type"] == "dependency_reduction":
                total_improvement += 25
        
        return f"{min(total_improvement, 70)}%"  # Cap at 70% improvement
    
    async def handle_message(self, message: Message) -> None:
        """Handle task manager specific messages."""
        self.logger.debug(f"Task manager received message: {message.message_type} from {message.sender}")
        
        if message.message_type == MessageType.TASK_ASSIGNMENT:
            content = message.content
            self.logger.debug(f"Task assignment content: {content}")
            
            if content.get("action") == "decompose_query":
                self.logger.info(f"Processing decompose_query request")
                
                # Create decomposition task
                task = Task(
                    title="Decompose research query",
                    description="Break down query into actionable subtasks",
                    task_type="decompose_query",
                    priority=TaskPriority.HIGH,
                    metadata={
                        "query_data": {
                            "query": content.get("query"),
                            "query_type": content.get("query_type"),
                            "parameters": content.get("parameters", {})
                        }
                    }
                )
                
                # Process immediately and respond
                try:
                    self.logger.info(f"Processing decomposition task")
                    result = await self.process_task(task)
                    
                    self.logger.info(f"Decomposition completed, sending response to {message.sender}")
                    
                    # Create response message with response_to field set
                    response_message = Message(
                        sender=self.config.agent_id,
                        recipient=message.sender,
                        message_type=MessageType.TASK_RESULT,
                        content=result,
                        task_id=task.id,
                        response_to=message.id
                    )
                    
                    # Send response via message bus
                    await self.message_bus.publish(f"agent.{message.sender}", response_message)
                    
                except Exception as e:
                    self.logger.error(f"Decomposition failed: {e}")
                    
                    # Create error response message
                    error_response = Message(
                        sender=self.config.agent_id,
                        recipient=message.sender,
                        message_type=MessageType.ERROR_REPORT,
                        content={"error": str(e)},
                        task_id=task.id,
                        response_to=message.id
                    )
                    
                    await self.message_bus.publish(f"agent.{message.sender}", error_response)
    
    def get_task_manager_status(self) -> Dict[str, Any]:
        """Get detailed task manager status."""
        status = self.get_status()
        status.update({
            "decomposition_templates": list(self.decomposition_templates.keys()),
            "agent_capabilities": self.agent_capabilities
        })
        return status

