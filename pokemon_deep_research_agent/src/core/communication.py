"""
Communication system for inter-agent messaging.

This module implements the shared task channel and message passing infrastructure
that enables coordination between different agents in the system.
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID

from .models import Message, MessageType, Task, TaskStatus


class MessageBus:
    """
    Central message bus for agent communication.

    Implements a publish-subscribe pattern with topic-based routing
    and message persistence for debugging and monitoring.
    """

    def __init__(self, max_message_history: int = 10000):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_history: deque = deque(maxlen=max_message_history)
        self.pending_responses: Dict[UUID, asyncio.Future] = {}
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self._running = False

    async def start(self) -> None:
        """Start the message bus."""
        self._running = True
        self.logger.info("Message bus started")

    async def stop(self) -> None:
        """Stop the message bus."""
        self._running = False
        # Cancel any pending responses
        for future in self.pending_responses.values():
            if not future.done():
                future.cancel()
        self.pending_responses.clear()
        self.logger.info("Message bus stopped")

    async def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to messages on a specific topic."""
        async with self._lock:
            self.subscribers[topic].append(callback)
            self.logger.debug(f"Agent subscribed to topic: {topic}")

    async def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Unsubscribe from messages on a specific topic."""
        async with self._lock:
            if callback in self.subscribers[topic]:
                self.subscribers[topic].remove(callback)
                self.logger.debug(f"Agent unsubscribed from topic: {topic}")

    async def publish(self, topic: str, message: Message) -> None:
        """Publish a message to all subscribers of a topic."""
        async with self._lock:
            self.message_history.append((datetime.now(), topic, message))

            # Notify all subscribers
            for callback in self.subscribers[topic]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    self.logger.error(f"Error in message callback: {e}")

            # Handle response messages
            if message.response_to and message.response_to in self.pending_responses:
                self.logger.debug(
                    f"Handling response message: {message.id} -> {message.response_to}"
                )
                future = self.pending_responses.pop(message.response_to)
                if not future.done():
                    future.set_result(message)
                    self.logger.debug(
                        f"Response future completed for {message.response_to}"
                    )
                else:
                    self.logger.warning(
                        f"Response future already done for {message.response_to}"
                    )
            elif message.response_to:
                self.logger.warning(
                    f"Received response {message.id} for unknown request {message.response_to}"
                )

    async def send_message(
        self,
        sender: str,
        recipient: str,
        message_type: MessageType,
        content: Dict[str, Any],
        task_id: Optional[UUID] = None,
        requires_response: bool = False,
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """Send a message and optionally wait for a response."""

        message = Message(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            content=content,
            task_id=task_id,
            requires_response=requires_response,
        )

        response_future = None
        if requires_response:
            response_future = asyncio.Future()
            self.pending_responses[message.id] = response_future

        # Publish to recipient's topic
        await self.publish(f"agent.{recipient}", message)

        # Also publish to general coordination topic
        await self.publish("coordination", message)

        if requires_response and response_future:
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                self.logger.warning(f"Message timeout: {message.id}")
                if message.id in self.pending_responses:
                    del self.pending_responses[message.id]
                return None

        return None

    async def broadcast(
        self,
        sender: str,
        message_type: MessageType,
        content: Dict[str, Any],
        task_id: Optional[UUID] = None,
    ) -> None:
        """Broadcast a message to all agents."""

        message = Message(
            sender=sender,
            recipient="*",
            message_type=message_type,
            content=content,
            task_id=task_id,
        )

        await self.publish("broadcast", message)

    def get_message_history(
        self,
        agent_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[tuple]:
        """Get message history with optional filtering."""

        filtered_messages = []
        for timestamp, topic, message in reversed(self.message_history):
            if since and timestamp < since:
                continue
            if (
                agent_id
                and message.sender != agent_id
                and message.recipient != agent_id
            ):
                continue
            if message_type and message.message_type != message_type:
                continue

            filtered_messages.append((timestamp, topic, message))
            if len(filtered_messages) >= limit:
                break

        return list(reversed(filtered_messages))

    async def update_agent_status(self, agent_id: str, status: Dict[str, Any]) -> None:
        """Update agent status information."""
        async with self._lock:
            self.agent_status[agent_id] = {**status, "last_updated": datetime.now()}

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an agent."""
        return self.agent_status.get(agent_id)

    def get_all_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents."""
        return self.agent_status.copy()

    @property
    def subscriptions(self) -> Dict[str, List[Callable]]:
        """Get all subscriptions."""
        return dict(self.subscribers)


class TaskChannel:
    """
    Shared task channel for coordinating work between agents.

    Implements the task distribution and dependency management system
    similar to the Eigent Workforce architecture.
    """

    def __init__(self, message_bus: Optional[MessageBus] = None):
        self.message_bus = message_bus or MessageBus()
        self.tasks: Dict[UUID, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.completed_tasks: Dict[UUID, Task] = {}
        self.agent_assignments: Dict[str, Set[UUID]] = defaultdict(set)
        self.task_dependencies: Dict[UUID, Set[UUID]] = defaultdict(set)
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self._running = False

    async def start(self) -> None:
        """Start the task channel."""
        self._running = True
        if not self.message_bus._running:
            await self.message_bus.start()
        self.logger.info("Task channel started")

    async def stop(self) -> None:
        """Stop the task channel."""
        self._running = False
        # Clear pending tasks
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.logger.info("Task channel stopped")

    async def submit_task(self, task: Task) -> None:
        """Submit a new task to the channel."""
        async with self._lock:
            self.tasks[task.id] = task

            # Check if dependencies are satisfied
            if await self._are_dependencies_satisfied(task.id):
                await self.task_queue.put(task)
                self.logger.info(f"Task submitted and queued: {task.title}")
            else:
                self.logger.info(
                    f"Task submitted but waiting for dependencies: {task.title}"
                )

        # Notify about new task
        await self.message_bus.broadcast(
            sender="task_channel",
            message_type=MessageType.TASK_ASSIGNMENT,
            content={
                "action": "task_submitted",
                "task_id": str(task.id),
                "task_title": task.title,
                "task_type": task.task_type,
            },
            task_id=task.id,
        )

    async def get_next_task(
        self, agent_id: str, agent_capabilities: List[str]
    ) -> Optional[Task]:
        """Get the next available task for an agent."""
        try:
            # Wait for a task with timeout
            task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

            self.logger.debug(f"Agent {agent_id} checking task: {task.title}")
            self.logger.debug(f"Agent capabilities: {agent_capabilities}")
            self.logger.debug(
                f"Task required capability: {task.metadata.get('required_capability', 'None')}"
            )

            # Check if agent can handle this task
            if self._can_agent_handle_task(task, agent_capabilities):
                async with self._lock:
                    task.assigned_agent = agent_id
                    task.update_status(TaskStatus.IN_PROGRESS)
                    self.agent_assignments[agent_id].add(task.id)

                self.logger.info(f"Task assigned to {agent_id}: {task.title}")
                return task
            else:
                # Put task back in queue
                await self.task_queue.put(task)
                self.logger.debug(
                    f"Agent {agent_id} cannot handle task {task.title}, putting back in queue"
                )
                return None

        except asyncio.TimeoutError:
            return None

    async def complete_task(
        self, task_id: UUID, result: Dict[str, Any], agent_id: str
    ) -> None:
        """Mark a task as completed and store the result."""
        async with self._lock:
            if task_id not in self.tasks:
                self.logger.error(f"Task not found: {task_id}")
                return

            task = self.tasks[task_id]
            task.result = result
            task.update_status(TaskStatus.COMPLETED)

            # Move to completed tasks
            self.completed_tasks[task_id] = task
            del self.tasks[task_id]

            # Remove from agent assignments
            if agent_id in self.agent_assignments:
                self.agent_assignments[agent_id].discard(task_id)

        # Check for dependent tasks that can now be executed
        await self._check_dependent_tasks(task_id)

        # Notify about task completion
        await self.message_bus.broadcast(
            sender="task_channel",
            message_type=MessageType.TASK_RESULT,
            content={
                "action": "task_completed",
                "task_id": str(task_id),
                "agent_id": agent_id,
                "result": result,
            },
            task_id=task_id,
        )

        self.logger.info(f"Task completed by {agent_id}: {task.title}")

    async def fail_task(self, task_id: UUID, error_message: str, agent_id: str) -> None:
        """Mark a task as failed."""
        async with self._lock:
            if task_id not in self.tasks:
                self.logger.error(f"Task not found: {task_id}")
                return

            task = self.tasks[task_id]
            task.update_status(TaskStatus.FAILED, error_message)

            # Remove from agent assignments
            if agent_id in self.agent_assignments:
                self.agent_assignments[agent_id].discard(task_id)

        # Notify about task failure
        await self.message_bus.broadcast(
            sender="task_channel",
            message_type=MessageType.ERROR_REPORT,
            content={
                "action": "task_failed",
                "task_id": str(task_id),
                "agent_id": agent_id,
                "error": error_message,
            },
            task_id=task_id,
        )

        self.logger.error(f"Task failed by {agent_id}: {task.title} - {error_message}")

    async def get_task_result(self, task_id: UUID) -> Optional[Dict[str, Any]]:
        """Get the result of a completed task."""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].result
        return None

    async def get_task_status(self, task_id: UUID) -> Optional[TaskStatus]:
        """Get the current status of a task."""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id].status
        return None

    def get_agent_tasks(self, agent_id: str) -> List[Task]:
        """Get all tasks currently assigned to an agent."""
        task_ids = self.agent_assignments.get(agent_id, set())
        return [self.tasks[task_id] for task_id in task_ids if task_id in self.tasks]

    async def _are_dependencies_satisfied(self, task_id: UUID) -> bool:
        """Check if all dependencies for a task are satisfied."""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True

    async def _check_dependent_tasks(self, completed_task_id: UUID) -> None:
        """Check if any pending tasks can now be executed."""
        tasks_to_queue = []

        async with self._lock:
            for task_id, task in self.tasks.items():
                if (
                    task.status == TaskStatus.PENDING
                    and completed_task_id in task.dependencies
                    and await self._are_dependencies_satisfied(task_id)
                ):
                    tasks_to_queue.append(task)

        for task in tasks_to_queue:
            await self.task_queue.put(task)
            self.logger.info(f"Task queued after dependency completion: {task.title}")

    def _can_agent_handle_task(self, task: Task, agent_capabilities: List[str]) -> bool:
        """Check if an agent can handle a specific task."""
        required_capability = task.metadata.get("required_capability")
        if required_capability and required_capability not in agent_capabilities:
            return False
        return True

    async def get_task_error(self, task_id: UUID) -> Optional[str]:
        """Get the error message of a failed task."""
        if task_id in self.tasks:
            return self.tasks[task_id].error_message
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id].error_message
        return None

    @property
    def pending_tasks(self) -> Dict[UUID, Task]:
        """Get all pending tasks."""
        return {
            task_id: task
            for task_id, task in self.tasks.items()
            if task.status == TaskStatus.PENDING
        }

    @property
    def subscriptions(self) -> Dict[str, List[Callable]]:
        """Get all message bus subscriptions (for MessageBus compatibility)."""
        return self.message_bus.subscribers if self.message_bus else {}
