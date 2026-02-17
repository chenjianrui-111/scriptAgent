"""
达人/商品基础数据仓储

目标:
  - 为 ProfileAgent / ProductAgent 提供数据库读取能力
  - 无记录时自动回退到现有 mock/规则逻辑
  - 默认使用 SQLite，方便本地与单机部署
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from script_agent.config.settings import settings
from script_agent.models.context import InfluencerProfile, ProductProfile, StyleProfile

logger = logging.getLogger(__name__)


def _loads_list(raw: str) -> List[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _dumps_list(values: List[str]) -> str:
    return json.dumps([str(v).strip() for v in values if str(v).strip()], ensure_ascii=False)


class DomainDataRepository:
    """SQLite 数据仓储（达人画像 + 商品画像）。"""

    def __init__(
        self,
        db_path: str = "",
        *,
        enabled: Optional[bool] = None,
        auto_init_schema: Optional[bool] = None,
    ):
        cfg = settings.domain_data
        self.enabled = cfg.enabled if enabled is None else bool(enabled)
        self.auto_init_schema = (
            cfg.auto_init_schema if auto_init_schema is None else bool(auto_init_schema)
        )
        self.db_path = Path(db_path or cfg.sqlite_path)
        self._schema_ready = False

    async def get_influencer_profile(
        self,
        *,
        influencer_id: str = "",
        name: str = "",
        category: str = "",
    ) -> Optional[InfluencerProfile]:
        if not self.enabled:
            return None
        return await asyncio.to_thread(
            self._get_influencer_profile_sync,
            influencer_id.strip(),
            name.strip(),
            category.strip(),
        )

    async def get_product_profile(
        self,
        *,
        product_id: str = "",
        name: str = "",
        category: str = "",
    ) -> Optional[ProductProfile]:
        if not self.enabled:
            return None
        return await asyncio.to_thread(
            self._get_product_profile_sync,
            product_id.strip(),
            name.strip(),
            category.strip(),
        )

    async def upsert_influencer_profile(self, profile: InfluencerProfile) -> None:
        if not self.enabled:
            return
        await asyncio.to_thread(self._upsert_influencer_profile_sync, profile)

    async def upsert_product_profile(self, product: ProductProfile) -> None:
        if not self.enabled:
            return
        await asyncio.to_thread(self._upsert_product_profile_sync, product)

    def _ensure_schema_sync(self) -> bool:
        if self._schema_ready:
            return True
        if not self.auto_init_schema and not self.db_path.exists():
            return False
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect_sync() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS influencer_profiles (
                        influencer_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL DEFAULT '',
                        category TEXT NOT NULL DEFAULT '',
                        platform TEXT NOT NULL DEFAULT '',
                        follower_count INTEGER NOT NULL DEFAULT 0,
                        audience_age_range TEXT NOT NULL DEFAULT '',
                        audience_gender_ratio TEXT NOT NULL DEFAULT '',
                        top_content_keywords TEXT NOT NULL DEFAULT '[]',
                        style_tone TEXT NOT NULL DEFAULT '',
                        style_formality_level REAL NOT NULL DEFAULT 0.5,
                        style_catchphrases TEXT NOT NULL DEFAULT '[]',
                        style_avg_sentence_length REAL NOT NULL DEFAULT 20.0,
                        style_punctuation_style TEXT NOT NULL DEFAULT 'normal',
                        style_selling_point_style TEXT NOT NULL DEFAULT 'direct',
                        style_interaction_frequency REAL NOT NULL DEFAULT 0.5,
                        style_humor_level REAL NOT NULL DEFAULT 0.3,
                        style_sample_count INTEGER NOT NULL DEFAULT 0,
                        style_confidence REAL NOT NULL DEFAULT 0.5,
                        style_version INTEGER NOT NULL DEFAULT 1
                    );
                    CREATE INDEX IF NOT EXISTS idx_influencer_name_category
                    ON influencer_profiles(name, category);

                    CREATE TABLE IF NOT EXISTS product_profiles (
                        product_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL DEFAULT '',
                        category TEXT NOT NULL DEFAULT '',
                        brand TEXT NOT NULL DEFAULT '',
                        price_range TEXT NOT NULL DEFAULT '',
                        features TEXT NOT NULL DEFAULT '[]',
                        selling_points TEXT NOT NULL DEFAULT '[]',
                        target_audience TEXT NOT NULL DEFAULT '',
                        compliance_notes TEXT NOT NULL DEFAULT '[]'
                    );
                    CREATE INDEX IF NOT EXISTS idx_product_name_category
                    ON product_profiles(name, category);
                    """
                )
                conn.commit()
            self._schema_ready = True
            return True
        except Exception as exc:
            logger.warning("domain data schema init failed: %s", exc)
            return False

    def _connect_sync(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _get_influencer_profile_sync(
        self,
        influencer_id: str,
        name: str,
        category: str,
    ) -> Optional[InfluencerProfile]:
        if not self._ensure_schema_sync():
            return None
        try:
            with self._connect_sync() as conn:
                row = None
                if influencer_id:
                    row = conn.execute(
                        "SELECT * FROM influencer_profiles WHERE influencer_id = ? LIMIT 1",
                        (influencer_id,),
                    ).fetchone()
                elif name:
                    if category:
                        row = conn.execute(
                            """
                            SELECT *
                            FROM influencer_profiles
                            WHERE name = ?
                              AND (category = ? OR category = '')
                            ORDER BY CASE WHEN category = ? THEN 0 ELSE 1 END
                            LIMIT 1
                            """,
                            (name, category, category),
                        ).fetchone()
                    else:
                        row = conn.execute(
                            """
                            SELECT *
                            FROM influencer_profiles
                            WHERE name = ?
                            LIMIT 1
                            """,
                            (name,),
                        ).fetchone()
        except Exception as exc:
            logger.warning("query influencer profile failed: %s", exc)
            return None

        if row is None:
            return None

        style = StyleProfile(
            influencer_id=row["influencer_id"] or "",
            tone=row["style_tone"] or "",
            formality_level=float(row["style_formality_level"] or 0.5),
            catchphrases=_loads_list(row["style_catchphrases"] or ""),
            avg_sentence_length=float(row["style_avg_sentence_length"] or 20.0),
            punctuation_style=row["style_punctuation_style"] or "normal",
            selling_point_style=row["style_selling_point_style"] or "direct",
            interaction_frequency=float(row["style_interaction_frequency"] or 0.5),
            humor_level=float(row["style_humor_level"] or 0.3),
            sample_count=int(row["style_sample_count"] or 0),
            confidence=float(row["style_confidence"] or 0.5),
            version=int(row["style_version"] or 1),
        )
        return InfluencerProfile(
            influencer_id=row["influencer_id"] or "",
            name=row["name"] or "",
            category=row["category"] or "",
            platform=row["platform"] or "",
            follower_count=int(row["follower_count"] or 0),
            style=style,
            audience_age_range=row["audience_age_range"] or "",
            audience_gender_ratio=row["audience_gender_ratio"] or "",
            top_content_keywords=_loads_list(row["top_content_keywords"] or ""),
        )

    def _get_product_profile_sync(
        self,
        product_id: str,
        name: str,
        category: str,
    ) -> Optional[ProductProfile]:
        if not self._ensure_schema_sync():
            return None
        try:
            with self._connect_sync() as conn:
                row = None
                if product_id:
                    row = conn.execute(
                        "SELECT * FROM product_profiles WHERE product_id = ? LIMIT 1",
                        (product_id,),
                    ).fetchone()
                elif name:
                    if category:
                        row = conn.execute(
                            """
                            SELECT *
                            FROM product_profiles
                            WHERE name = ?
                              AND (category = ? OR category = '')
                            ORDER BY CASE WHEN category = ? THEN 0 ELSE 1 END
                            LIMIT 1
                            """,
                            (name, category, category),
                        ).fetchone()
                    else:
                        row = conn.execute(
                            """
                            SELECT *
                            FROM product_profiles
                            WHERE name = ?
                            LIMIT 1
                            """,
                            (name,),
                        ).fetchone()
        except Exception as exc:
            logger.warning("query product profile failed: %s", exc)
            return None

        if row is None:
            return None

        return ProductProfile(
            product_id=row["product_id"] or "",
            name=row["name"] or "",
            category=row["category"] or "",
            brand=row["brand"] or "",
            price_range=row["price_range"] or "",
            features=_loads_list(row["features"] or ""),
            selling_points=_loads_list(row["selling_points"] or ""),
            target_audience=row["target_audience"] or "",
            compliance_notes=_loads_list(row["compliance_notes"] or ""),
        )

    def _upsert_influencer_profile_sync(self, profile: InfluencerProfile) -> None:
        if not self._ensure_schema_sync():
            return
        with self._connect_sync() as conn:
            conn.execute(
                """
                INSERT INTO influencer_profiles (
                    influencer_id, name, category, platform, follower_count,
                    audience_age_range, audience_gender_ratio, top_content_keywords,
                    style_tone, style_formality_level, style_catchphrases,
                    style_avg_sentence_length, style_punctuation_style,
                    style_selling_point_style, style_interaction_frequency,
                    style_humor_level, style_sample_count, style_confidence, style_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(influencer_id) DO UPDATE SET
                    name = excluded.name,
                    category = excluded.category,
                    platform = excluded.platform,
                    follower_count = excluded.follower_count,
                    audience_age_range = excluded.audience_age_range,
                    audience_gender_ratio = excluded.audience_gender_ratio,
                    top_content_keywords = excluded.top_content_keywords,
                    style_tone = excluded.style_tone,
                    style_formality_level = excluded.style_formality_level,
                    style_catchphrases = excluded.style_catchphrases,
                    style_avg_sentence_length = excluded.style_avg_sentence_length,
                    style_punctuation_style = excluded.style_punctuation_style,
                    style_selling_point_style = excluded.style_selling_point_style,
                    style_interaction_frequency = excluded.style_interaction_frequency,
                    style_humor_level = excluded.style_humor_level,
                    style_sample_count = excluded.style_sample_count,
                    style_confidence = excluded.style_confidence,
                    style_version = excluded.style_version
                """,
                (
                    profile.influencer_id,
                    profile.name,
                    profile.category,
                    profile.platform,
                    int(profile.follower_count),
                    profile.audience_age_range,
                    profile.audience_gender_ratio,
                    _dumps_list(profile.top_content_keywords),
                    profile.style.tone,
                    float(profile.style.formality_level),
                    _dumps_list(profile.style.catchphrases),
                    float(profile.style.avg_sentence_length),
                    profile.style.punctuation_style,
                    profile.style.selling_point_style,
                    float(profile.style.interaction_frequency),
                    float(profile.style.humor_level),
                    int(profile.style.sample_count),
                    float(profile.style.confidence),
                    int(profile.style.version),
                ),
            )
            conn.commit()

    def _upsert_product_profile_sync(self, product: ProductProfile) -> None:
        if not self._ensure_schema_sync():
            return
        with self._connect_sync() as conn:
            conn.execute(
                """
                INSERT INTO product_profiles (
                    product_id, name, category, brand, price_range,
                    features, selling_points, target_audience, compliance_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_id) DO UPDATE SET
                    name = excluded.name,
                    category = excluded.category,
                    brand = excluded.brand,
                    price_range = excluded.price_range,
                    features = excluded.features,
                    selling_points = excluded.selling_points,
                    target_audience = excluded.target_audience,
                    compliance_notes = excluded.compliance_notes
                """,
                (
                    product.product_id,
                    product.name,
                    product.category,
                    product.brand,
                    product.price_range,
                    _dumps_list(product.features),
                    _dumps_list(product.selling_points),
                    product.target_audience,
                    _dumps_list(product.compliance_notes),
                ),
            )
            conn.commit()
