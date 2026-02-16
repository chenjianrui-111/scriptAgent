"""
测试套件 - 核心Agent组件测试

运行: cd script_agent && python -m pytest tests/ -v
"""

import asyncio
import pytest
import sys
import os

# 确保项目根目录在path中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =======================================================================
# 1. 模型测试
# =======================================================================

class TestModels:
    """数据模型测试"""

    def test_agent_message(self):
        from script_agent.models.message import AgentMessage, MessageType
        msg = AgentMessage(
            trace_id="test-123",
            source="orchestrator",
            target="intent",
            payload={"query": "帮我写一段美妆话术"},
        )
        assert msg.trace_id == "test-123"
        assert msg.message_type == MessageType.REQUEST

        resp = msg.create_response({"result": "ok"})
        assert resp.parent_message_id == msg.message_id
        assert resp.message_type == MessageType.RESPONSE

    def test_intent_result(self):
        from script_agent.models.message import IntentResult
        result = IntentResult(
            intent="script_generation",
            confidence=0.85,
            slots={"category": "美妆", "scenario": "直播带货"},
        )
        assert result.intent == "script_generation"
        assert result.confidence == 0.85
        assert result.slots["category"] == "美妆"

    def test_session_context(self):
        from script_agent.models.context import SessionContext
        session = SessionContext(session_id="s1", tenant_id="t1")
        turn = session.add_turn(
            user_message="帮我写话术",
            assistant_message="好的，生成如下...",
        )
        assert len(session.turns) == 1
        assert turn.turn_index == 0

    def test_entity_cache(self):
        from script_agent.models.context import EntityCache
        cache = EntityCache()
        cache.update("influencer", "inf_001", "小雅")
        result = cache.get_latest("influencer")
        assert result is not None
        assert result["name"] == "小雅"


# =======================================================================
# 2. 状态机测试
# =======================================================================

class TestStateMachine:
    """状态机测试"""

    def test_normal_flow(self):
        from script_agent.models.state_machine import StateMachine, StateContext, WorkflowState
        sm = StateMachine()
        ctx = StateContext()

        # INIT → CONTEXT_LOADING
        sm.transition(ctx, {})
        assert ctx.current_state == WorkflowState.CONTEXT_LOADING

        # CONTEXT_LOADING → INTENT_RECOGNIZING
        sm.transition(ctx, {})
        assert ctx.current_state == WorkflowState.INTENT_RECOGNIZING

        # INTENT_RECOGNIZING → PROFILE_FETCHING (高置信)
        sm.transition(ctx, {"confidence": 0.85})
        assert ctx.current_state == WorkflowState.PROFILE_FETCHING

    def test_clarification_flow(self):
        from script_agent.models.state_machine import StateMachine, StateContext, WorkflowState
        sm = StateMachine()
        ctx = StateContext()
        sm.transition(ctx, {})  # → CONTEXT_LOADING
        sm.transition(ctx, {})  # → INTENT_RECOGNIZING

        # 低置信 → 澄清
        sm.transition(ctx, {"confidence": 0.3})
        assert ctx.current_state == WorkflowState.INTENT_CLARIFYING

    def test_quality_retry(self):
        from script_agent.models.state_machine import StateMachine, StateContext, WorkflowState
        sm = StateMachine()
        ctx = StateContext(current_state=WorkflowState.QUALITY_CHECKING)

        # 质量不通过 + 可重试
        result = sm.transition(ctx, {
            "quality_passed": False, "retry_count": 1, "max_retries": 3
        })
        assert ctx.current_state == WorkflowState.REGENERATING


# =======================================================================
# 3. 意图识别Agent测试
# =======================================================================

class TestIntentAgent:
    """意图识别测试"""

    def test_slot_extraction(self):
        from script_agent.agents.intent_agent import SlotExtractor
        extractor = SlotExtractor()

        slots = extractor.extract("帮我写一段美妆直播开场话术", "script_generation")
        assert slots.get("category") == "美妆"
        assert slots.get("scenario") == "直播带货"
        assert slots.get("sub_scenario") == "开场话术"

    def test_slot_extraction_event(self):
        from script_agent.agents.intent_agent import SlotExtractor
        extractor = SlotExtractor()
        slots = extractor.extract("618活动的促销话术", "")
        assert slots.get("event") == "618大促"

    def test_slot_extraction_style(self):
        from script_agent.agents.intent_agent import SlotExtractor
        extractor = SlotExtractor()
        slots = extractor.extract("写一段活泼幽默的话术", "")
        assert "lively" in slots.get("style_hint", "")
        assert "humorous" in slots.get("style_hint", "")

    def test_slot_extraction_product(self):
        from script_agent.agents.intent_agent import SlotExtractor

        extractor = SlotExtractor()
        slots = extractor.extract(
            "帮我写这款小金瓶精华的直播话术，卖点：成分安全、提亮肤色",
            "script_generation",
        )
        assert "小金瓶精华" in slots.get("product_name", "")
        assert "成分安全" in slots.get("selling_points", [])

    def test_reference_resolver(self):
        from script_agent.agents.intent_agent import ReferenceResolver
        from script_agent.models.context import SessionContext, EntityCache

        session = SessionContext()
        session.entity_cache.update("influencer", "inf_001", "小雅")

        resolver = ReferenceResolver()
        resolved = resolver.resolve("帮这个达人也写一段", session)
        assert resolved.get("target_name") == "小雅"

    def test_continuation_resolver(self):
        from script_agent.agents.intent_agent import ReferenceResolver
        from script_agent.models.context import SessionContext

        session = SessionContext()
        session.slot_context.update("script_generation", {
            "category": "美妆", "scenario": "直播带货"
        })

        resolver = ReferenceResolver()
        resolved = resolver.resolve("再来一段", session)
        assert resolved.get("category") == "美妆"
        assert resolved.get("_continuation") is True

    def test_intent_clarifier(self):
        from script_agent.agents.intent_agent import IntentClarifier
        clarifier = IntentClarifier()

        # 缺少必填槽位
        result = clarifier.check("script_generation", {"category": "美妆"})
        assert result is not None  # 需要澄清 scenario

        # 槽位完整
        result = clarifier.check("script_generation", {
            "category": "美妆", "scenario": "直播带货"
        })
        assert result is None  # 无需澄清


# =======================================================================
# 4. 质量校验Agent测试
# =======================================================================

class TestQualityAgent:
    """质量校验测试"""

    def test_sensitive_word_check(self):
        from script_agent.agents.quality_agent import SensitiveWordChecker
        checker = SensitiveWordChecker()

        passed, issues = checker.check("这是最好的产品")
        assert not passed
        assert any(i["word"] == "最好" for i in issues)

        passed, issues = checker.check("这款产品非常不错")
        assert passed

    def test_compliance_check(self):
        from script_agent.agents.quality_agent import ComplianceChecker
        checker = ComplianceChecker()

        passed, issues = checker.check("保证7天见效，无效退款")
        assert not passed

    def test_style_consistency(self):
        from script_agent.agents.quality_agent import StyleConsistencyChecker
        from script_agent.models.context import InfluencerProfile, StyleProfile

        checker = StyleConsistencyChecker()
        profile = InfluencerProfile(
            style=StyleProfile(
                tone="活泼", catchphrases=["宝子们"],
                formality_level=0.3, avg_sentence_length=15,
                interaction_frequency=0.7,
            )
        )

        score, suggestions = checker.check(
            "宝子们！今天给大家推荐一款超好用的面膜呀！你们觉得怎么样？",
            profile,
        )
        assert score >= 0.7  # 风格匹配度应该较高


# =======================================================================
# 5. 上下文压缩测试
# =======================================================================

class TestContextCompression:
    """上下文压缩测试"""

    def test_importance_scorer(self):
        from script_agent.context.session_compressor import TurnImportanceScorer
        from script_agent.models.context import SessionContext, ConversationTurn

        scorer = TurnImportanceScorer()
        session = SessionContext()

        # 包含话术生成的轮次应该更重要
        turn_with_script = ConversationTurn(
            turn_index=0,
            user_message="写话术",
            assistant_message="话术内容...",
            generated_script="宝子们！...",
        )
        session.turns = [turn_with_script]
        score = scorer.score(turn_with_script, session)
        assert score > 0.5

    def test_rule_compressor(self):
        from script_agent.context.session_compressor import RuleCompressor
        from script_agent.models.context import ConversationTurn

        compressor = RuleCompressor()
        turn = ConversationTurn(
            user_message="帮我写一段美妆直播开场话术，要求风格活泼接地气有感染力" * 5,  # 长消息(>80字)
            assistant_message="宝子们晚上好！" * 20,
            generated_script="宝子们晚上好！...",
        )
        compressed = compressor.compress_turn(turn)
        assert len(compressed["user"]) < len(turn.user_message)
        assert "[已生成话术]" in compressed["assistant"]


# =======================================================================
# 6. 风格提取测试
# =======================================================================

class TestStyleExtractor:
    """风格提取测试"""

    def test_catchphrase_extraction(self):
        from script_agent.services.style_extractor import CatchphraseExtractor

        extractor = CatchphraseExtractor()
        texts = [
            "宝子们晚上好！今天给大家推荐一款绝了的口红！",
            "宝子们！这个真的太好用了！绝了绝了！",
            "家人们看过来！今天宝子们有福了！",
        ]
        results = extractor.extract(texts)
        phrases = [r.phrase for r in results]
        assert "宝子们" in phrases

    def test_style_extraction(self):
        from script_agent.services.style_extractor import StyleExtractor

        extractor = StyleExtractor()
        texts = [
            "宝子们晚上好呀！今天给大家推荐的这款面膜真的绝了！上脸超级舒服啊！你们觉得怎么样？",
            "姐妹们！这个口红必入！显白到不行！真的爱了爱了！快来抢哦～",
        ]
        style = extractor.extract_from_texts(texts)
        assert style.formality_level < 0.5    # 口语化
        assert len(style.catchphrases) > 0
        assert style.sample_count == 2

    def test_profile_update_merge(self):
        from script_agent.services.style_extractor import ProfileUpdater, ExtractedStyle
        from script_agent.models.context import StyleProfile

        updater = ProfileUpdater()
        existing = StyleProfile(
            influencer_id="inf_001",
            tone="活泼", formality_level=0.3,
            catchphrases=["宝子们"], confidence=0.7,
        )
        extracted = ExtractedStyle(
            tone="专业", formality_level=0.6,
            catchphrases=["家人们", "宝子们"],
            sample_count=5, confidence=0.6,
        )
        merged = updater.merge(existing, extracted)
        assert merged.version == existing.version + 1
        assert "宝子们" in merged.catchphrases
        assert "家人们" in merged.catchphrases
        # 加权融合: 0.3*0.7 + 0.6*0.3 = 0.39
        assert 0.35 < merged.formality_level < 0.45


# =======================================================================
# 7. 集成测试 (Orchestrator)
# =======================================================================

class TestOrchestrator:
    """编排器集成测试 (不依赖外部LLM服务)"""

    def test_orchestrator_init(self):
        from script_agent.agents.orchestrator import Orchestrator
        orch = Orchestrator()
        assert orch.intent_agent is not None
        assert orch.profile_agent is not None
        assert orch.script_agent is not None
        assert orch.quality_agent is not None

    def test_session_manager(self):
        from script_agent.services.session_manager import SessionManager
        sm = SessionManager()

        async def _test():
            session = await sm.create(tenant_id="t1", influencer_name="小雅")
            assert session.session_id
            loaded = await sm.load(session.session_id)
            assert loaded is not None
            assert loaded.influencer_name == "小雅"

        asyncio.run(_test())

    def test_orchestrator_resume_seed(self):
        from script_agent.agents.orchestrator import Orchestrator
        from script_agent.models.context import SessionContext

        session = SessionContext(session_id="s1")
        session.workflow_snapshot = {
            "status": "in_progress",
            "last_query": "帮我写开场话术",
            "intent": {
                "intent": "script_generation",
                "confidence": 0.9,
                "slots": {"category": "美妆", "scenario": "直播带货"},
            },
            "retry_count": 1,
        }

        orch = Orchestrator()
        seed = orch._build_resume_seed("帮我写开场话术", session.workflow_snapshot)
        assert seed.get("intent_result") is not None
        assert seed["intent_result"].intent == "script_generation"
        assert seed["retry_count"] == 1


class TestEnterpriseFeatures:
    """企业级能力测试: 恢复机制 + 并发管理"""

    def test_session_snapshot_serialization(self):
        from script_agent.models.context import SessionContext
        from script_agent.services.session_manager import SessionSerializer

        session = SessionContext(session_id="s1", tenant_id="t1")
        session.workflow_snapshot = {
            "status": "in_progress",
            "current_state": "PROFILE_FETCHING",
            "last_query": "继续生成",
        }

        serializer = SessionSerializer()
        data = serializer.to_dict(session)
        restored = serializer.from_dict(data)
        assert restored.workflow_snapshot["status"] == "in_progress"
        assert restored.workflow_snapshot["current_state"] == "PROFILE_FETCHING"

    def test_session_lock_manager_serializes_same_session(self):
        from script_agent.services.concurrency import SessionLockManager

        async def _test():
            manager = SessionLockManager(default_timeout_seconds=2.0)
            events = []

            async def worker(name: str, delay: float):
                async with manager.acquire("session-1"):
                    events.append(f"{name}-start")
                    await asyncio.sleep(delay)
                    events.append(f"{name}-end")

            await asyncio.gather(
                worker("a", 0.05),
                worker("b", 0.01),
            )
            # 串行保证：一个 worker 的 end 必须先于另一个 worker 的 start
            serialized = (
                events.index("a-end") < events.index("b-start")
                or events.index("b-end") < events.index("a-start")
            )
            assert serialized

        asyncio.run(_test())

    def test_session_lock_manager_stats(self):
        from script_agent.services.concurrency import SessionLockManager

        async def _test():
            manager = SessionLockManager(default_timeout_seconds=1.0)
            stats = await manager.stats()
            assert "tracked_sessions" in stats
            assert "active_locks" in stats
            assert "queued_waiters" in stats

        asyncio.run(_test())

    def test_orchestrator_dedup_cache_hit(self):
        from script_agent.agents.orchestrator import Orchestrator
        from datetime import datetime

        snapshot = {
            "status": "completed",
            "last_query": "同一请求",
            "updated_at": datetime.now().isoformat(),
            "script": {
                "script_id": "x1",
                "content": "缓存话术",
                "content_truncated": False,
                "category": "美妆",
                "scenario": "直播带货",
            },
            "intent": {
                "intent": "script_generation",
                "confidence": 0.95,
                "slots": {"category": "美妆", "scenario": "直播带货"},
            },
            "state_history": ["COMPLETED"],
        }

        orch = Orchestrator()
        result = orch._build_cached_completed_result(
            snapshot=snapshot,
            query="同一请求",
            trace_id="t1",
        )
        assert result is not None
        assert result["from_cache"] is True
        assert result["script"].content == "缓存话术"

    def test_orchestrator_checkpoint_truncate_recovery_safe(self):
        from script_agent.agents.orchestrator import Orchestrator
        from script_agent.models.context import SessionContext
        from script_agent.models.message import GeneratedScript

        session = SessionContext(session_id="s1")
        orch = Orchestrator()

        async def _test():
            await orch._write_checkpoint(
                {
                    "session": session,
                    "status": "in_progress",
                    "trace_id": "t",
                    "query": "q",
                    "current_state": "SCRIPT_GENERATING",
                    "state_history": ["SCRIPT_GENERATING"],
                    "retry_count": 0,
                    "script": GeneratedScript(content="a" * 5000),
                },
                status="in_progress",
            )

        asyncio.run(_test())
        seed = orch._build_resume_seed("q", session.workflow_snapshot)
        # 截断脚本不参与恢复，避免半内容复用
        assert "script" not in seed

    def test_checkpoint_manager_memory_versioning_replay(self):
        from script_agent.services.checkpoint_store import (
            MemoryCheckpointStore,
            WorkflowCheckpointManager,
        )

        async def _test():
            manager = WorkflowCheckpointManager(store=MemoryCheckpointStore())
            r1 = await manager.write("s1", {"current_state": "A"}, "t1", "in_progress")
            r2 = await manager.write("s1", {"current_state": "B"}, "t2", "completed")
            assert r1["version"] == 1
            assert r2["version"] == 2

            latest = await manager.latest_payload("s1")
            assert latest is not None
            assert latest["current_state"] == "B"

            replay = await manager.replay("s1", 1)
            assert replay is not None
            assert replay["current_state"] == "A"

            history = await manager.history("s1", limit=10)
            assert len(history) == 2
            assert history[0]["version"] == 2

        asyncio.run(_test())

    def test_core_rate_limiter_local(self):
        from script_agent.services.core_rate_limiter import LocalCoreRateLimiter

        async def _test():
            limiter = LocalCoreRateLimiter(qps_per_tenant=2, tokens_per_minute=10)
            allow1 = await limiter.check_and_consume("t1", token_cost=3)
            allow2 = await limiter.check_and_consume("t1", token_cost=3)
            deny_qps = await limiter.check_and_consume("t1", token_cost=3)
            assert allow1.allowed
            assert allow2.allowed
            assert not deny_qps.allowed
            assert deny_qps.reason == "qps_limit_exceeded"

        asyncio.run(_test())

    def test_session_manager_retention_policy(self):
        from script_agent.config.settings import settings
        from script_agent.models.context import SessionContext
        from script_agent.services.session_manager import SessionManager

        async def _test():
            manager = SessionManager()
            session = SessionContext(session_id="s-retain", tenant_id="t1")
            for i in range(settings.context.max_turns_persisted + 5):
                session.add_turn(
                    user_message=("用户需求很长 " + str(i)) * 20,
                    assistant_message=("系统回复很长 " + str(i)) * 20,
                )

            await manager.save(session)
            loaded = await manager.load("s-retain")
            assert loaded is not None
            assert len(loaded.turns) <= settings.context.max_turns_persisted
            if len(loaded.turns) > settings.context.zone_a_turns:
                assert any(
                    t.is_compressed
                    for t in loaded.turns[:-settings.context.zone_a_turns]
                )
            await manager.close()

        asyncio.run(_test())

    def test_longterm_memory_recall_local(self):
        from script_agent.models.context import InfluencerProfile, ProductProfile, SessionContext
        from script_agent.models.message import GeneratedScript, IntentResult
        from script_agent.services.long_term_memory import (
            HashEmbeddingProvider,
            LongTermMemoryRetriever,
            MemoryVectorStore,
            RecallQuery,
        )

        async def _test():
            retriever = LongTermMemoryRetriever(
                store=MemoryVectorStore(),
                embedder=HashEmbeddingProvider(dim=128),
            )
            session = SessionContext(session_id="s-memory", tenant_id="tenant-x")
            profile = InfluencerProfile(influencer_id="inf-x", category="美妆")
            product = ProductProfile(
                product_id="p1",
                name="小金瓶精华",
                category="美妆",
                selling_points=["提亮肤色", "成分安全"],
            )
            script = GeneratedScript(
                content="姐妹们这款小金瓶精华上脸吸收快，提亮很明显，成分也很安心。",
                category="美妆",
                scenario="直播带货",
            )
            await retriever.remember_script(
                session=session,
                intent=IntentResult(intent="script_generation", confidence=0.9),
                profile=profile,
                product=product,
                script=script,
                query="直播介绍小金瓶精华",
            )
            hits = await retriever.recall(
                RecallQuery(
                    text="小金瓶精华 直播卖点 提亮 成分安全",
                    tenant_id="tenant-x",
                    influencer_id="inf-x",
                    category="美妆",
                    product_name="小金瓶精华",
                    top_k=3,
                )
            )
            assert len(hits) >= 1
            assert "小金瓶精华" in hits[0].get("text", "")
            await retriever.close()

        asyncio.run(_test())

    def test_product_agent_builds_profile(self):
        from script_agent.agents.product_agent import ProductAgent
        from script_agent.models.context import InfluencerProfile, SessionContext
        from script_agent.services.long_term_memory import (
            HashEmbeddingProvider,
            LongTermMemoryRetriever,
            MemoryVectorStore,
        )

        async def _test():
            retriever = LongTermMemoryRetriever(
                store=MemoryVectorStore(),
                embedder=HashEmbeddingProvider(dim=128),
            )
            agent = ProductAgent(memory=retriever)
            profile = InfluencerProfile(influencer_id="inf-1", category="美妆")
            session = SessionContext(session_id="s-product", tenant_id="t1")
            product, _hits = await agent.fetch(
                {
                    "category": "美妆",
                    "product_name": "小金瓶精华",
                    "selling_points": ["成分安全", "提亮肤色"],
                },
                profile=profile,
                session=session,
                query="帮我写小金瓶精华卖点文案",
            )
            assert product.name == "小金瓶精华"
            assert "成分安全" in product.selling_points
            assert product.category == "美妆"
            await retriever.close()

        asyncio.run(_test())


# =======================================================================
# Run
# =======================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
