"""
Base agent class for the Pokémon Deep Research Agent system.

This module defines the abstract base class that all specialized agents inherit from,
providing common functionality for task processing, communication, and lifecycle management.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..core.communication import MessageBus, TaskChannel
from ..core.models import (AgentConfig, AgentType, Message, MessageType, Task,
                           TaskStatus)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    Provides common functionality for task processing, inter-agent communication,
    error handling, and lifecycle management.
    """

    def __init__(
        self, config: AgentConfig, message_bus: MessageBus, task_channel: TaskChannel
    ):
        self.config = config
        self.message_bus = message_bus
        self.task_channel = task_channel
        self.logger = logging.getLogger(f"{__name__}.{config.agent_id}")

        # Agent state
        self.is_running = False
        self.current_tasks: Dict[UUID, Task] = {}
        self.completed_tasks_count = 0
        self.failed_tasks_count = 0
        self.start_time: Optional[datetime] = None

        # Performance metrics
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0

        # Task processing semaphore
        self.task_semaphore = asyncio.Semaphore(config.max_concurrent_tasks)

    async def start(self) -> None:
        """Start the agent and begin processing tasks."""
        if self.is_running:
            self.logger.warning("Agent is already running")
            return

        self.is_running = True
        self.start_time = datetime.now()

        # Subscribe to messages
        await self.message_bus.subscribe(
            f"agent.{self.config.agent_id}", self._handle_message
        )
        await self.message_bus.subscribe("broadcast", self._handle_broadcast)

        # Initialize agent-specific resources
        await self._initialize()

        # Start task processing loop
        asyncio.create_task(self._task_processing_loop())

        # Update status
        await self._update_status()

        self.logger.info(f"Agent {self.config.agent_id} started")

    async def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        if not self.is_running:
            return

        self.is_running = False

        # Wait for current tasks to complete
        await self._wait_for_current_tasks()

        # Cleanup agent-specific resources
        await self._cleanup()

        # Unsubscribe from messages
        await self.message_bus.unsubscribe(
            f"agent.{self.config.agent_id}", self._handle_message
        )
        await self.message_bus.unsubscribe("broadcast", self._handle_broadcast)

        self.logger.info(f"Agent {self.config.agent_id} stopped")

    async def _task_processing_loop(self) -> None:
        """Main task processing loop."""
        while self.is_running:
            try:
                # Get next available task
                task = await self.task_channel.get_next_task(
                    self.config.agent_id, [cap.name for cap in self.config.capabilities]
                )

                if task:
                    # Process task asynchronously
                    asyncio.create_task(self._process_task_with_semaphore(task))
                else:
                    # No tasks available, wait a bit
                    await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(1.0)

    async def _process_task_with_semaphore(self, task: Task) -> None:
        """Process a task with concurrency control."""
        async with self.task_semaphore:
            await self._process_task_wrapper(task)

    async def _process_task_wrapper(self, task: Task) -> None:
        """Wrapper for task processing with error handling and metrics."""
        start_time = datetime.now()

        try:
            self.current_tasks[task.id] = task
            self.logger.info(f"Starting task: {task.title}")

            # Process the task
            result = await self.process_task(task)

            # Complete the task
            await self.task_channel.complete_task(task.id, result, self.config.agent_id)

            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, success=True)

            self.logger.info(f"Completed task: {task.title} in {processing_time:.2f}s")

        except Exception as e:
            # Handle task failure
            error_message = f"Task processing failed: {str(e)}"
            await self.task_channel.fail_task(
                task.id, error_message, self.config.agent_id
            )

            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, success=False)

            self.logger.error(f"Failed task: {task.title} - {error_message}")

        finally:
            # Remove from current tasks
            if task.id in self.current_tasks:
                del self.current_tasks[task.id]

            # Update status
            await self._update_status()

    @abstractmethod
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        Process a task and return the result.

        This method must be implemented by each specialized agent.

        Args:
            task: The task to process

        Returns:
            Dict containing the task result

        Raises:
            Exception: If task processing fails
        """
        pass

    async def send_message(
        self,
        recipient: str,
        message_type: MessageType,
        content: Dict[str, Any],
        task_id: Optional[UUID] = None,
        requires_response: bool = False,
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """Send a message to another agent."""
        return await self.message_bus.send_message(
            sender=self.config.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            task_id=task_id,
            requires_response=requires_response,
            timeout=timeout,
        )

    async def broadcast_message(
        self,
        message_type: MessageType,
        content: Dict[str, Any],
        task_id: Optional[UUID] = None,
    ) -> None:
        """Broadcast a message to all agents."""
        await self.message_bus.broadcast(
            sender=self.config.agent_id,
            message_type=message_type,
            content=content,
            task_id=task_id,
        )

    async def _handle_message(self, message: Message) -> None:
        """Handle incoming messages."""
        try:
            if message.sender == self.config.agent_id:
                return  # Ignore own messages

            self.logger.debug(
                f"Received message from {message.sender}: {message.message_type}"
            )

            # Handle different message types
            if message.message_type == MessageType.QUERY_CLARIFICATION:
                await self._handle_query_clarification(message)
            elif message.message_type == MessageType.STATUS_UPDATE:
                await self._handle_status_update(message)
            elif message.message_type == MessageType.COORDINATION:
                await self._handle_coordination_message(message)
            elif message.message_type == MessageType.TASK_ASSIGNMENT:
                # Let subclasses handle task assignment messages
                await self.handle_message(message)
            elif message.message_type == MessageType.TASK_RESULT:
                # Let subclasses handle task result messages
                await self.handle_message(message)
            elif message.message_type == MessageType.ERROR_REPORT:
                # Let subclasses handle error report messages
                await self.handle_message(message)
            else:
                # Let subclasses handle other message types
                await self.handle_message(message)

        except Exception as e:
            self.logger.error(f"Error handling message: {e}")

    async def _handle_broadcast(self, message: Message) -> None:
        """Handle broadcast messages."""
        if message.sender != self.config.agent_id:
            await self._handle_message(message)

    async def handle_message(self, message: Message) -> None:
        """
        Handle agent-specific messages.

        Override this method in subclasses to handle specialized message types.
        """
        pass

    async def _handle_query_clarification(self, message: Message) -> None:
        """Handle query clarification requests."""
        # Default implementation - subclasses can override
        pass

    async def _handle_status_update(self, message: Message) -> None:
        """Handle status update messages."""
        # Default implementation - subclasses can override
        pass

    async def _handle_coordination_message(self, message: Message) -> None:
        """Handle coordination messages."""
        # Default implementation - subclasses can override
        pass

    async def _initialize(self) -> None:
        """Initialize agent-specific resources. Override in subclasses."""
        pass

    async def _cleanup(self) -> None:
        """Cleanup agent-specific resources. Override in subclasses."""
        pass

    async def _wait_for_current_tasks(self) -> None:
        """Wait for all current tasks to complete."""
        if not self.current_tasks:
            return

        self.logger.info(f"Waiting for {len(self.current_tasks)} tasks to complete...")

        # Wait up to 30 seconds for tasks to complete
        for _ in range(300):  # 30 seconds with 0.1s intervals
            if not self.current_tasks:
                break
            await asyncio.sleep(0.1)

        if self.current_tasks:
            self.logger.warning(
                f"Stopping with {len(self.current_tasks)} tasks still running"
            )

    def _update_metrics(self, processing_time: float, success: bool) -> None:
        """Update performance metrics."""
        self.total_processing_time += processing_time

        if success:
            self.completed_tasks_count += 1
        else:
            self.failed_tasks_count += 1

        total_tasks = self.completed_tasks_count + self.failed_tasks_count
        if total_tasks > 0:
            self.average_processing_time = self.total_processing_time / total_tasks

    async def _update_status(self) -> None:
        """Update agent status in the message bus."""
        uptime = 0.0
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        status = {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type.value,
            "is_running": self.is_running,
            "current_tasks": len(self.current_tasks),
            "completed_tasks": self.completed_tasks_count,
            "failed_tasks": self.failed_tasks_count,
            "uptime_seconds": uptime,
            "average_processing_time": self.average_processing_time,
            "capabilities": [cap.name for cap in self.config.capabilities],
        }

        await self.message_bus.update_agent_status(self.config.agent_id, status)

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        uptime = 0.0
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type.value,
            "name": self.config.name,
            "description": self.config.description,
            "is_running": self.is_running,
            "current_tasks": len(self.current_tasks),
            "completed_tasks": self.completed_tasks_count,
            "failed_tasks": self.failed_tasks_count,
            "uptime_seconds": uptime,
            "average_processing_time": self.average_processing_time,
            "capabilities": [cap.name for cap in self.config.capabilities],
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
        }
