import asyncio

from script_agent.agents.product_agent import ProductAgent
from script_agent.agents.profile_agent import ProfileAgent
from script_agent.models.context import (
    InfluencerProfile,
    ProductProfile,
    SessionContext,
    StyleProfile,
)
from script_agent.models.message import AgentMessage, GeneratedScript, IntentResult, QualityResult
from script_agent.services.domain_data_repository import DomainDataRepository
from script_agent.services.long_term_memory import (
    HashEmbeddingProvider,
    LongTermMemoryRetriever,
    MemoryVectorStore,
)
from script_agent.skills.base import SkillContext
from script_agent.skills.builtin.script_gen import ScriptGenerationSkill


def test_domain_data_repository_roundtrip(tmp_path):
    async def _run():
        repo = DomainDataRepository(db_path=str(tmp_path / "domain_data.db"))
        profile = InfluencerProfile(
            influencer_id="inf-db-1",
            name="数据库达人",
            category="美妆",
            platform="douyin",
            follower_count=100000,
            style=StyleProfile(
                influencer_id="inf-db-1",
                tone="专业亲和",
                formality_level=0.45,
                catchphrases=["姐妹们"],
                avg_sentence_length=18,
                humor_level=0.3,
                interaction_frequency=0.65,
                confidence=0.88,
            ),
            audience_age_range="18-30",
            audience_gender_ratio="女80%",
            top_content_keywords=["成分党", "抗老", "修护"],
        )
        product = ProductProfile(
            product_id="prod-db-1",
            name="玻尿酸精华液",
            category="美妆",
            brand="某某品牌",
            price_range="199-299",
            features=["高保湿", "轻薄易吸收"],
            selling_points=["维稳修护", "提亮肤感"],
            target_audience="干皮/混干",
            compliance_notes=["避免绝对化描述"],
        )

        await repo.upsert_influencer_profile(profile)
        await repo.upsert_product_profile(product)

        loaded_profile = await repo.get_influencer_profile(
            influencer_id="inf-db-1",
            category="美妆",
        )
        loaded_product = await repo.get_product_profile(
            product_id="prod-db-1",
            category="美妆",
        )

        assert loaded_profile is not None
        assert loaded_profile.name == "数据库达人"
        assert loaded_profile.style.tone == "专业亲和"
        assert "成分党" in loaded_profile.top_content_keywords

        assert loaded_product is not None
        assert loaded_product.brand == "某某品牌"
        assert "维稳修护" in loaded_product.selling_points
        assert loaded_product.target_audience == "干皮/混干"

    asyncio.run(_run())


def test_profile_agent_prefers_database_record(tmp_path):
    async def _run():
        repo = DomainDataRepository(db_path=str(tmp_path / "domain_data.db"))
        await repo.upsert_influencer_profile(
            InfluencerProfile(
                influencer_id="inf-db-2",
                name="落库达人",
                category="食品",
                style=StyleProfile(tone="生活化", catchphrases=["家人们"]),
            )
        )
        agent = ProfileAgent(repository=repo)
        msg = AgentMessage(
            payload={
                "slots": {
                    "target_id": "inf-db-2",
                    "target_name": "落库达人",
                    "category": "食品",
                }
            }
        )
        resp = await agent(msg)
        profile = resp.payload["profile"]
        assert profile.influencer_id == "inf-db-2"
        assert profile.name == "落库达人"
        assert profile.style.tone == "生活化"

    asyncio.run(_run())


def test_product_agent_merges_database_and_slot_data(tmp_path):
    async def _run():
        repo = DomainDataRepository(db_path=str(tmp_path / "domain_data.db"))
        await repo.upsert_product_profile(
            ProductProfile(
                product_id="prod-db-2",
                name="小金瓶精华",
                category="美妆",
                brand="库内品牌",
                price_range="159-239",
                features=["修护", "易吸收"],
                selling_points=["提亮肤色", "稳定肤感"],
                target_audience="混油皮",
                compliance_notes=["避免医疗功效描述"],
            )
        )

        retriever = LongTermMemoryRetriever(
            store=MemoryVectorStore(),
            embedder=HashEmbeddingProvider(dim=128),
        )
        agent = ProductAgent(memory=retriever, repository=repo)
        profile = InfluencerProfile(influencer_id="inf-1", category="美妆")
        session = SessionContext(session_id="s-product-db", tenant_id="t1")
        product, _hits = await agent.fetch(
            {
                "product_id": "prod-db-2",
                "category": "美妆",
                "selling_points": ["直播专享赠品"],
                "product_features": ["成分温和"],
            },
            profile=profile,
            session=session,
            query="给我一段小金瓶精华直播话术",
        )

        assert product.name == "小金瓶精华"
        assert product.brand == "库内品牌"
        assert "直播专享赠品" in product.selling_points
        assert "提亮肤色" in product.selling_points
        assert "成分温和" in product.features
        assert product.target_audience == "混油皮"
        await retriever.close()

    asyncio.run(_run())


def test_quality_agent_retry_path_is_effective():
    class FakeScriptAgent:
        def __init__(self):
            self.feedback_calls = []

        async def __call__(self, message: AgentMessage):
            feedback = list(message.payload.get("feedback", []))
            self.feedback_calls.append(feedback)
            if feedback:
                content = "宝子们今天这款精华我二刷实测过，吸收速度和润感都在线，互动区告诉我你们最关心保湿还是提亮。"
            else:
                content = "宝子们上链接前先听我说，今天这款精华主打温和和修护，我先把成分、肤感和使用场景三个重点讲透，再给你们直播福利。"
            return message.create_response(
                payload={"script": GeneratedScript(content=content)},
                source="script_generation",
            )

    class FakeQualityAgent:
        def __init__(self):
            self.calls = 0

        async def __call__(self, message: AgentMessage):
            self.calls += 1
            if self.calls == 1:
                result = QualityResult(
                    passed=False,
                    overall_score=0.58,
                    style_consistency=0.4,
                    suggestions=["建议增加互动引导（提问/点赞引导）"],
                )
            else:
                result = QualityResult(
                    passed=True,
                    overall_score=0.91,
                    style_consistency=0.88,
                    suggestions=[],
                )
            return message.create_response(
                payload={"quality_result": result},
                source="quality_check",
            )

    async def _run():
        skill = ScriptGenerationSkill()
        fake_script = FakeScriptAgent()
        fake_quality = FakeQualityAgent()
        skill._script_agent = fake_script
        skill._quality_agent = fake_quality

        context = SkillContext(
            intent=IntentResult(
                intent="script_generation",
                confidence=0.95,
                slots={"category": "美妆", "scenario": "开场话术"},
            ),
            profile=InfluencerProfile(category="美妆"),
            session=SessionContext(session_id="s-quality-check"),
            trace_id="trace-quality-check",
        )

        result = await skill.execute(context)
        assert result.success is True
        assert result.quality_result is not None
        assert result.quality_result.passed is True
        assert fake_quality.calls == 2
        assert fake_script.feedback_calls[0] == []
        assert "互动引导" in fake_script.feedback_calls[1][0]

    asyncio.run(_run())
