"""
Coordinator Agent for the Pokémon Deep Research Agent system.

The Coordinator Agent acts as the "project manager" of the system, responsible for:
- Receiving and parsing user queries
- Routing tasks to appropriate specialized agents
- Monitoring task progress and handling failures
- Coordinating inter-agent communication
- Managing workflow state and transitions
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..agents.base_agent import BaseAgent
from ..core.communication import MessageBus, TaskChannel
from ..core.models import (
    AgentConfig, AgentType, Message, MessageType, ResearchQuery, 
    Task, TaskPriority, TaskStatus
)
from ..tools.llm_client import LLMClient


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent that manages the overall research workflow.
    
    This agent serves as the central orchestrator, receiving user queries,
    coordinating with the Task Manager for decomposition, and routing
    tasks to appropriate specialized agents.
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
        
        # Active research sessions
        self.active_queries: Dict[UUID, ResearchQuery] = {}
        self.query_tasks: Dict[UUID, List[UUID]] = {}  # Query ID -> Task IDs
        self.task_results: Dict[UUID, Dict[str, Any]] = {}
        
        # Agent registry and capabilities
        self.available_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # Workflow state
        self.pending_clarifications: Dict[UUID, Dict[str, Any]] = {}
        
    async def _initialize(self) -> None:
        """Initialize coordinator-specific resources."""
        # Subscribe to agent status updates
        await self.message_bus.subscribe("agent_status", self._handle_agent_status)
        
        self.logger.info("Coordinator Agent initialized")
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process coordinator-specific tasks."""
        task_type = task.task_type
        
        if task_type == "research_query":
            return await self._handle_research_query(task)
        elif task_type == "query_clarification":
            return await self._handle_query_clarification(task)
        elif task_type == "workflow_coordination":
            return await self._handle_workflow_coordination(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _handle_research_query(self, task: Task) -> Dict[str, Any]:
        """Handle a new research query from the user."""
        query_data = task.metadata.get("query_data", {})
        original_query = query_data.get("query", "")
        
        if not original_query:
            raise ValueError("No query provided in task metadata")
        
        # Create research query object
        research_query = ResearchQuery(
            original_query=original_query,
            query_type="unknown",  # Will be determined by analysis
            parameters=query_data.get("parameters", {}),
            user_preferences=query_data.get("preferences", {})
        )
        
        self.active_queries[research_query.id] = research_query
        
        # Analyze the query using LLM
        query_analysis = {}  # Initialize with default value
        try:
            query_analysis = await self.llm_client.analyze_pokemon_query(original_query)
            research_query.query_type = query_analysis.get("query_type", "general")
            research_query.parameters.update(query_analysis)
            
            self.logger.info(f"Analyzed query: {original_query} -> Type: {research_query.query_type}")
            
        except Exception as e:
            self.logger.error(f"Query analysis failed: {e}")
            # Continue with basic classification and default analysis
            research_query.query_type = "general"
            query_analysis = {
                "query_type": "general",
                "pokemon_mentioned": [],
                "game_context": "general",
                "battle_format": "unknown",
                "difficulty_level": "intermediate",
                "constraints": []
            }
        
        # Check if clarification is needed
        if await self._needs_clarification(research_query, query_analysis):
            return await self._request_clarification(research_query, query_analysis)
        
        # Proceed with task decomposition
        return await self._initiate_research_workflow(research_query)
    
    async def _needs_clarification(
        self, 
        query: ResearchQuery, 
        analysis: Dict[str, Any]
    ) -> bool:
        """Determine if the query needs clarification."""
        # Check for ambiguous or incomplete information
        ambiguity_indicators = [
            analysis.get("game_context") == "unknown",
            analysis.get("battle_format") == "unknown",
            analysis.get("difficulty_level") == "unknown",
            len(analysis.get("pokemon_mentioned", [])) == 0 and query.query_type == "individual_analysis",
            "ambiguous" in analysis.get("notes", "").lower()
        ]
        
        # Require clarification if multiple indicators are present
        return sum(ambiguity_indicators) >= 2
    
    async def _request_clarification(
        self, 
        query: ResearchQuery, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Request clarification from the user."""
        try:
            clarification_questions = await self.llm_client.generate_clarification_questions(
                query.original_query, analysis
            )
            
            # Store pending clarification
            self.pending_clarifications[query.id] = {
                "questions": clarification_questions,
                "analysis": analysis,
                "timestamp": datetime.now()
            }
            
            return {
                "status": "clarification_needed",
                "query_id": str(query.id),
                "questions": clarification_questions,
                "message": "I need some additional information to provide the best research results."
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate clarification questions: {e}")
            # Proceed without clarification
            return await self._initiate_research_workflow(query)
    
    async def _handle_query_clarification(self, task: Task) -> Dict[str, Any]:
        """Handle user responses to clarification questions."""
        clarification_data = task.metadata.get("clarification_data", {})
        query_id = UUID(clarification_data.get("query_id"))
        user_responses = clarification_data.get("responses", {})
        
        if query_id not in self.pending_clarifications:
            raise ValueError(f"No pending clarification for query {query_id}")
        
        # Update query with clarification
        query = self.active_queries[query_id]
        clarification_info = self.pending_clarifications[query_id]
        
        # Update query parameters based on responses
        query.parameters.update(user_responses)
        query.clarified_query = self._build_clarified_query(query, user_responses)
        
        # Remove from pending clarifications
        del self.pending_clarifications[query_id]
        
        # Proceed with research workflow
        return await self._initiate_research_workflow(query)
    
    def _build_clarified_query(
        self, 
        query: ResearchQuery, 
        responses: Dict[str, Any]
    ) -> str:
        """Build a clarified query string from user responses."""
        clarified = query.original_query
        
        # Add context from responses
        if "game_version" in responses:
            clarified += f" (for {responses['game_version']})"
        
        if "battle_format" in responses:
            clarified += f" in {responses['battle_format']} format"
        
        if "difficulty_level" in responses:
            clarified += f" for {responses['difficulty_level']} players"
        
        return clarified
    
    async def _initiate_research_workflow(self, query: ResearchQuery) -> Dict[str, Any]:
        """Initiate the research workflow by coordinating with Task Manager."""
        # Send task decomposition request to Task Manager
        query_text = query.clarified_query or query.original_query
        response = await self.send_message(
            recipient="task_manager",
            message_type=MessageType.TASK_ASSIGNMENT,
            content={
                "action": "decompose_query",
                "query": query_text,
                "query_type": query.query_type,
                "parameters": query.parameters
            },
            task_id=query.id,
            requires_response=True
        )
        
        if not response:
            raise Exception("Task Manager did not respond to decomposition request")
        
        # Extract task list from response
        task_list = response.content.get("tasks", [])
        
        if not task_list:
            raise Exception("Task Manager returned empty task list")
        
        # Create and submit tasks
        task_ids = []
        for i, task_info in enumerate(task_list):
            task = Task(
                title=task_info.get("title", "Research Task"),
                description=task_info.get("description", ""),
                task_type=task_info.get("task_type", "research"),
                priority=TaskPriority(task_info.get("priority", 2)),
                metadata={
                    "query_id": str(query.id),
                    "required_capability": task_info.get("required_capability"),
                    "task_parameters": task_info.get("parameters", {})
                }
            )
            
            self.logger.info(f"Creating subtask {i+1}: {task.title}")
            self.logger.info(f"  Task type: {task.task_type}")
            self.logger.info(f"  Required capability: {task_info.get('required_capability')}")
            
            await self.task_channel.submit_task(task)
            task_ids.append(task.id)
        
        # Track tasks for this query
        self.query_tasks[query.id] = task_ids
        
        self.logger.info(f"Initiated research workflow for query {query.id} with {len(task_ids)} tasks")
        
        return {
            "status": "research_initiated",
            "query_id": str(query.id),
            "tasks_created": len(task_ids),
            "estimated_completion_time": self._estimate_completion_time(task_list)
        }
    
    def _estimate_completion_time(self, task_list: List[Dict[str, Any]]) -> str:
        """Estimate completion time based on task complexity."""
        base_time = 30  # Base 30 seconds
        complex_tasks = sum(1 for task in task_list if task.get("complexity", "medium") == "high")
        medium_tasks = sum(1 for task in task_list if task.get("complexity", "medium") == "medium")
        
        estimated_seconds = base_time + (complex_tasks * 60) + (medium_tasks * 30)
        
        if estimated_seconds < 60:
            return f"{estimated_seconds} seconds"
        elif estimated_seconds < 3600:
            return f"{estimated_seconds // 60} minutes"
        else:
            return f"{estimated_seconds // 3600} hours"
    
    async def _handle_workflow_coordination(self, task: Task) -> Dict[str, Any]:
        """Handle workflow coordination tasks."""
        coordination_type = task.metadata.get("coordination_type")
        
        if coordination_type == "check_progress":
            return await self._check_research_progress(task)
        elif coordination_type == "synthesize_results":
            return await self._synthesize_research_results(task)
        else:
            raise ValueError(f"Unknown coordination type: {coordination_type}")
    
    async def _check_research_progress(self, task: Task) -> Dict[str, Any]:
        """Check progress of active research queries."""
        query_id = UUID(task.metadata.get("query_id"))
        
        if query_id not in self.query_tasks:
            return {"status": "query_not_found"}
        
        task_ids = self.query_tasks[query_id]
        completed_tasks = 0
        failed_tasks = 0
        
        for task_id in task_ids:
            status = await self.task_channel.get_task_status(task_id)
            if status == TaskStatus.COMPLETED:
                completed_tasks += 1
            elif status == TaskStatus.FAILED:
                failed_tasks += 1
        
        progress = {
            "total_tasks": len(task_ids),
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "in_progress": len(task_ids) - completed_tasks - failed_tasks,
            "completion_percentage": (completed_tasks / len(task_ids)) * 100
        }
        
        return {"status": "progress_checked", "progress": progress}
    
    async def _synthesize_research_results(self, task: Task) -> Dict[str, Any]:
        """Synthesize results from completed research tasks."""
        query_id = UUID(task.metadata.get("query_id"))
        
        if query_id not in self.active_queries:
            raise ValueError(f"Query {query_id} not found")
        
        query = self.active_queries[query_id]
        task_ids = self.query_tasks.get(query_id, [])
        
        # Collect results from completed tasks
        results = []
        for task_id in task_ids:
            result = await self.task_channel.get_task_result(task_id)
            if result:
                results.append(result)
        
        if not results:
            return {"status": "no_results", "message": "No completed tasks found"}
        
        # Use LLM to synthesize results
        try:
            # Convert results to ResearchResult objects for synthesis
            research_results = []
            for i, result in enumerate(results):
                research_results.append(type('ResearchResult', (), {
                    'query_id': query_id,
                    'agent_id': f"agent_{i}",
                    'result_type': result.get('type', 'general'),
                    'data': result,
                    'confidence': result.get('confidence', 1.0),
                    'sources': result.get('sources', []),
                    'timestamp': datetime.now(),
                    'metadata': {}
                })())
            
            synthesis = await self.llm_client.synthesize_research_findings(
                query, research_results
            )
            
            # Mark query as completed
            query.status = TaskStatus.COMPLETED
            
            return {
                "status": "synthesis_completed",
                "query_id": str(query_id),
                "synthesis": synthesis,
                "results_count": len(results)
            }
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            return {
                "status": "synthesis_failed",
                "error": str(e),
                "raw_results": results
            }
    
    async def _handle_task_completion(self, message: Message) -> None:
        """Handle task completion notifications."""
        task_id = UUID(message.content.get("task_id"))
        
        # Find which query this task belongs to
        query_id = None
        for qid, task_ids in self.query_tasks.items():
            if task_id in task_ids:
                query_id = qid
                break
        
        if not query_id:
            return
        
        # Check if all tasks for this query are completed
        task_ids = self.query_tasks[query_id]
        all_completed = True
        
        for tid in task_ids:
            status = await self.task_channel.get_task_status(tid)
            if status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                all_completed = False
                break
        
        if all_completed:
            # Trigger synthesis
            synthesis_task = Task(
                title=f"Synthesize results for query {query_id}",
                description="Synthesize research findings into final report",
                task_type="workflow_coordination",
                priority=TaskPriority.HIGH,
                metadata={
                    "coordination_type": "synthesize_results",
                    "query_id": str(query_id)
                }
            )
            
            await self.task_channel.submit_task(synthesis_task)
    
    async def _handle_task_failure(self, message: Message) -> None:
        """Handle task failure notifications."""
        task_id = UUID(message.content.get("task_id"))
        error_message = message.content.get("error", "Unknown error")
        
        # Find which query this task belongs to
        query_id = None
        for qid, task_ids in self.query_tasks.items():
            if task_id in task_ids:
                query_id = qid
                break
        
        if not query_id:
            return
        
        self.logger.warning(f"Task {task_id} failed for query {query_id}: {error_message}")
        
        # Check if all tasks for this query are completed (including failed ones)
        task_ids = self.query_tasks[query_id]
        all_completed = True
        
        for tid in task_ids:
            status = await self.task_channel.get_task_status(tid)
            if status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                all_completed = False
                break
        
        if all_completed:
            # Trigger synthesis even with some failed tasks
            synthesis_task = Task(
                title=f"Synthesize results for query {query_id}",
                description="Synthesize research findings into final report",
                task_type="workflow_coordination",
                priority=TaskPriority.HIGH,
                metadata={
                    "coordination_type": "synthesize_results",
                    "query_id": str(query_id)
                }
            )
            
            await self.task_channel.submit_task(synthesis_task)
    
    async def handle_message(self, message: Message) -> None:
        """Handle coordinator-specific messages."""
        self.logger.debug(f"Coordinator received message: {message.message_type} from {message.sender}")
        
        if message.message_type == MessageType.STATUS_UPDATE:
            await self._handle_agent_status_update(message)
        elif message.message_type == MessageType.COORDINATION:
            await self._handle_coordination_request(message)
        elif message.message_type == MessageType.TASK_RESULT:
            # Handle task completion notifications
            self.logger.debug(f"Received TASK_RESULT: {message.content}")
            if message.content.get("action") == "task_completed":
                await self._handle_task_completion(message)
        elif message.message_type == MessageType.ERROR_REPORT:
            # Handle task failure notifications
            if message.content.get("action") == "task_failed":
                await self._handle_task_failure(message)
    
    async def _handle_agent_status_update(self, message: Message) -> None:
        """Handle agent status updates."""
        agent_id = message.content.get("agent_id")
        status = message.content.get("status", {})
        
        if agent_id:
            self.available_agents[agent_id] = status
            self.agent_capabilities[agent_id] = status.get("capabilities", [])
    
    async def _handle_coordination_request(self, message: Message) -> None:
        """Handle coordination requests from other agents."""
        request_type = message.content.get("request_type")
        
        if request_type == "agent_registry":
            # Respond with available agents
            await self.send_message(
                recipient=message.sender,
                message_type=MessageType.COORDINATION,
                content={
                    "response_type": "agent_registry",
                    "available_agents": self.available_agents,
                    "agent_capabilities": self.agent_capabilities
                },
                response_to=message.id
            )
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get detailed coordinator status."""
        status = self.get_status()
        status.update({
            "active_queries": len(self.active_queries),
            "pending_clarifications": len(self.pending_clarifications),
            "available_agents": len(self.available_agents),
            "total_tasks_tracked": sum(len(tasks) for tasks in self.query_tasks.values())
        })
        return status


    
    async def _handle_agent_status(self, message: Message) -> None:
        """Handle agent status update messages."""
        try:
            agent_id = message.sender
            status_data = message.content
            
            # Update agent registry
            self.available_agents[agent_id] = {
                "status": status_data.get("status", "unknown"),
                "current_tasks": status_data.get("current_tasks", 0),
                "completed_tasks": status_data.get("completed_tasks", 0),
                "last_updated": datetime.now()
            }
            
            # Update capabilities if provided
            if "capabilities" in status_data:
                self.agent_capabilities[agent_id] = status_data["capabilities"]
            
            self.logger.debug(f"Updated status for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling agent status update: {e}")
    
    async def _handle_task_completion(self, message: Message) -> None:
        """Handle task completion notifications."""
        try:
            task_id_str = message.content.get("task_id")
            if not task_id_str:
                return
            
            task_id = UUID(task_id_str)
            result = message.content.get("result", {})
            
            # Store task result
            self.task_results[task_id] = result
            
            # Check if this completes any research queries
            await self._check_query_completion(task_id)
            
            self.logger.debug(f"Processed task completion for {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling task completion: {e}")
    
    async def _check_query_completion(self, completed_task_id: UUID) -> None:
        """Check if a completed task finishes any research queries."""
        for query_id, task_ids in self.query_tasks.items():
            if completed_task_id in task_ids:
                # Check if all tasks for this query are complete
                all_complete = True
                for task_id in task_ids:
                    task_status = await self.task_channel.get_task_status(task_id)
                    if task_status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        all_complete = False
                        break
                
                if all_complete:
                    await self._finalize_research_query(query_id)
    
    async def _finalize_research_query(self, query_id: UUID) -> None:
        """Finalize a completed research query."""
        try:
            if query_id not in self.active_queries:
                return
            
            research_query = self.active_queries[query_id]
            task_ids = self.query_tasks.get(query_id, [])
            
            # Collect all results
            all_results = []
            for task_id in task_ids:
                result = self.task_results.get(task_id)
                if result:
                    all_results.append(result)
            
            # Generate final synthesis (simplified for now)
            final_result = {
                "query_id": str(query_id),
                "original_query": research_query.original_query,
                "results": all_results,
                "synthesis": {
                    "synthesis_text": f"Research completed for: {research_query.original_query}",
                    "key_insights": [
                        {"insight": "Research analysis completed", "importance": "medium"}
                    ],
                    "data_quality": {"assessment": "good", "quality_score": 0.8}
                },
                "confidence": 0.8,
                "sources": ["multi_agent_research"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Store final result
            self.task_results[query_id] = final_result
            
            # Update query status
            research_query.status = TaskStatus.COMPLETED
            
            self.logger.info(f"Research query completed: {query_id}")
            
        except Exception as e:
            self.logger.error(f"Error finalizing research query: {e}")
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator-specific statistics."""
        status = self.get_status()
        status.update({
            "active_queries": len(self.active_queries),
            "available_agents": len(self.available_agents),
            "pending_clarifications": len(self.pending_clarifications),
            "total_task_results": len(self.task_results)
        })
        return status

