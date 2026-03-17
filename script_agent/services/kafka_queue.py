"""
Kafka异步任务队列服务

提供基于Kafka的异步任务处理能力，解耦API响应与话术生成。
- API立即返回session_id (P99 < 200ms)
- 后台Worker异步处理任务
- 支持任务状态查询和结果获取
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, asdict

from pydantic import BaseModel

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("aiokafka not installed. Kafka queue will not be available.")


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"      # 已提交，等待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"        # 失败
    TIMEOUT = "timeout"      # 超时


class TaskPriority(str, Enum):
    """任务优先级"""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class TaskMessage:
    """Kafka任务消息"""
    task_id: str
    session_id: str
    user_id: str
    task_type: str  # "generate_script", "batch_generate", etc.
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()

    def to_json(self) -> str:
        """序列化为JSON"""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "TaskMessage":
        """从JSON反序列化"""
        return cls(**json.loads(data))


class TaskResult(BaseModel):
    """任务结果"""
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None


class KafkaTaskQueue:
    """Kafka异步任务队列"""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic_prefix: str = "scriptagent",
        group_id: str = "scriptagent-workers",
        result_store: Optional[Any] = None,  # Redis/Memory store for task results
    ):
        if not KAFKA_AVAILABLE:
            raise ImportError("aiokafka is required for Kafka queue. Install with: pip install aiokafka")

        self.bootstrap_servers = bootstrap_servers
        self.topic_prefix = topic_prefix
        self.group_id = group_id
        self.result_store = result_store

        self.producer: Optional[AIOKafkaProducer] = None
        self.consumer: Optional[AIOKafkaConsumer] = None
        self._running = False
        self._task_handlers: Dict[str, Callable] = {}

    async def start_producer(self):
        """启动生产者"""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: v.encode('utf-8'),
            compression_type="gzip",
        )
        await self.producer.start()
        logger.info(f"Kafka producer started: {self.bootstrap_servers}")

    async def stop_producer(self):
        """停止生产者"""
        if self.producer:
            await self.producer.stop()
            logger.info("Kafka producer stopped")

    async def submit_task(
        self,
        task_id: str,
        session_id: str,
        user_id: str,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> str:
        """
        提交任务到Kafka队列

        Returns:
            task_id: 任务ID，用于查询状态
        """
        if not self.producer:
            raise RuntimeError("Producer not started. Call start_producer() first.")

        # 创建任务消息
        message = TaskMessage(
            task_id=task_id,
            session_id=session_id,
            user_id=user_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
        )

        # 根据优先级选择topic
        topic = self._get_topic_for_priority(priority)

        # 发送到Kafka
        await self.producer.send_and_wait(
            topic,
            value=message.to_json(),
            key=session_id.encode('utf-8'),  # 相同session的任务进入同一分区，保证顺序
        )

        # 初始化任务状态
        await self._save_task_status(task_id, TaskStatus.PENDING)

        logger.info(f"Task submitted: {task_id}, type={task_type}, priority={priority}")
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """查询任务状态"""
        if not self.result_store:
            return None

        # 从Redis/Memory读取任务结果
        key = f"task:result:{task_id}"
        data = await self.result_store.get(key)

        if not data:
            return None

        return TaskResult.parse_raw(data)

    def register_handler(self, task_type: str, handler: Callable):
        """注册任务处理器"""
        self._task_handlers[task_type] = handler
        logger.info(f"Registered handler for task_type: {task_type}")

    async def start_consumer(self, num_workers: int = 3):
        """启动消费者（Worker）"""
        if not self._task_handlers:
            raise RuntimeError("No task handlers registered. Call register_handler() first.")

        # 订阅所有优先级的topic
        topics = [
            self._get_topic_for_priority(p)
            for p in TaskPriority
        ]

        self.consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: v.decode('utf-8'),
        )

        await self.consumer.start()
        self._running = True
        logger.info(f"Kafka consumer started: topics={topics}, workers={num_workers}")

        # 启动多个Worker协程
        workers = [
            asyncio.create_task(self._worker_loop(worker_id))
            for worker_id in range(num_workers)
        ]

        await asyncio.gather(*workers)

    async def stop_consumer(self):
        """停止消费者"""
        self._running = False
        if self.consumer:
            await self.consumer.stop()
            logger.info("Kafka consumer stopped")

    async def _worker_loop(self, worker_id: int):
        """Worker主循环"""
        logger.info(f"Worker-{worker_id} started")

        async for msg in self.consumer:
            if not self._running:
                break

            try:
                # 解析任务消息
                task_msg = TaskMessage.from_json(msg.value)

                logger.info(
                    f"Worker-{worker_id} processing task: {task_msg.task_id}, "
                    f"type={task_msg.task_type}"
                )

                # 更新状态为处理中
                await self._save_task_status(task_msg.task_id, TaskStatus.PROCESSING)
                started_at = datetime.utcnow()

                # 调用处理器
                handler = self._task_handlers.get(task_msg.task_type)
                if not handler:
                    raise ValueError(f"No handler for task_type: {task_msg.task_type}")

                result = await handler(task_msg)

                # 保存结果
                completed_at = datetime.utcnow()
                processing_time = int((completed_at - started_at).total_seconds() * 1000)

                task_result = TaskResult(
                    task_id=task_msg.task_id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    started_at=started_at,
                    completed_at=completed_at,
                    processing_time_ms=processing_time,
                )

                await self._save_task_result(task_result)

                logger.info(
                    f"Worker-{worker_id} completed task: {task_msg.task_id}, "
                    f"time={processing_time}ms"
                )

            except Exception as e:
                logger.error(f"Worker-{worker_id} error: {e}", exc_info=True)

                # 保存失败状态
                if 'task_msg' in locals():
                    task_result = TaskResult(
                        task_id=task_msg.task_id,
                        status=TaskStatus.FAILED,
                        error=str(e),
                        completed_at=datetime.utcnow(),
                    )
                    await self._save_task_result(task_result)

        logger.info(f"Worker-{worker_id} stopped")

    def _get_topic_for_priority(self, priority: TaskPriority) -> str:
        """根据优先级获取topic名称"""
        return f"{self.topic_prefix}.tasks.{priority.value}"

    async def _save_task_status(self, task_id: str, status: TaskStatus):
        """保存任务状态"""
        if not self.result_store:
            return

        key = f"task:status:{task_id}"
        await self.result_store.set(key, status.value, ex=3600)  # 1小时过期

    async def _save_task_result(self, result: TaskResult):
        """保存任务结果"""
        if not self.result_store:
            return

        key = f"task:result:{result.task_id}"
        await self.result_store.set(
            key,
            result.json(),
            ex=86400,  # 24小时过期
        )


# 全局队列实例（懒加载）
_queue_instance: Optional[KafkaTaskQueue] = None


def get_kafka_queue() -> KafkaTaskQueue:
    """获取全局Kafka队列实例"""
    global _queue_instance
    if _queue_instance is None:
        raise RuntimeError("Kafka queue not initialized. Call init_kafka_queue() first.")
    return _queue_instance


def init_kafka_queue(
    bootstrap_servers: str,
    result_store: Any,
    **kwargs
) -> KafkaTaskQueue:
    """初始化全局Kafka队列"""
    global _queue_instance
    _queue_instance = KafkaTaskQueue(
        bootstrap_servers=bootstrap_servers,
        result_store=result_store,
        **kwargs
    )
    return _queue_instance
