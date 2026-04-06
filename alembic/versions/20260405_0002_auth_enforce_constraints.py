"""enforce user scoped constraints and not-null user_id

Revision ID: 20260405_0002
Revises: 20260405_0001
Create Date: 2026-04-05 01:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "20260405_0002"
down_revision: Union[str, Sequence[str], None] = "20260405_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


LEGACY_USER_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    # 1) 兜底回填，确保后续 not null 不失败
    op.execute(sa.text("UPDATE categories SET user_id = CAST(:uid AS uuid) WHERE user_id IS NULL").bindparams(uid=LEGACY_USER_ID))
    op.execute(sa.text("UPDATE sessions SET user_id = CAST(:uid AS uuid) WHERE user_id IS NULL").bindparams(uid=LEGACY_USER_ID))
    op.execute(sa.text("UPDATE documents SET user_id = CAST(:uid AS uuid) WHERE user_id IS NULL").bindparams(uid=LEGACY_USER_ID))
    op.execute(sa.text("UPDATE diary_entries SET user_id = CAST(:uid AS uuid) WHERE user_id IS NULL").bindparams(uid=LEGACY_USER_ID))

    # 2) 改为 NOT NULL
    op.alter_column("categories", "user_id", existing_type=postgresql.UUID(as_uuid=False), nullable=False)
    op.alter_column("sessions", "user_id", existing_type=postgresql.UUID(as_uuid=False), nullable=False)
    op.alter_column("documents", "user_id", existing_type=postgresql.UUID(as_uuid=False), nullable=False)
    op.alter_column("diary_entries", "user_id", existing_type=postgresql.UUID(as_uuid=False), nullable=False)

    # 3) 删除旧的全局唯一约束（name/date）
    op.execute("ALTER TABLE categories DROP CONSTRAINT IF EXISTS categories_name_key")
    op.execute("ALTER TABLE diary_entries DROP CONSTRAINT IF EXISTS diary_entries_date_key")

    # 4) 清理可能存在的唯一索引，重建为非唯一普通索引
    op.execute("DROP INDEX IF EXISTS ix_categories_name")
    op.execute("DROP INDEX IF EXISTS ix_diary_entries_date")

    op.create_index("ix_categories_name", "categories", ["name"], unique=False)
    op.create_index("ix_diary_entries_date", "diary_entries", ["date"], unique=False)

    # 5) 新增用户维度联合唯一
    op.create_unique_constraint("uq_categories_user_name", "categories", ["user_id", "name"])
    op.create_unique_constraint("uq_diary_entries_user_date", "diary_entries", ["user_id", "date"])


def downgrade() -> None:
    # 1) 去掉联合唯一
    op.drop_constraint("uq_diary_entries_user_date", "diary_entries", type_="unique")
    op.drop_constraint("uq_categories_user_name", "categories", type_="unique")

    # 2) 恢复旧的全局唯一约束
    op.create_unique_constraint("categories_name_key", "categories", ["name"])
    op.create_unique_constraint("diary_entries_date_key", "diary_entries", ["date"])

    # 3) user_id 退回可空
    op.alter_column("diary_entries", "user_id", existing_type=postgresql.UUID(as_uuid=False), nullable=True)
    op.alter_column("documents", "user_id", existing_type=postgresql.UUID(as_uuid=False), nullable=True)
    op.alter_column("sessions", "user_id", existing_type=postgresql.UUID(as_uuid=False), nullable=True)
    op.alter_column("categories", "user_id", existing_type=postgresql.UUID(as_uuid=False), nullable=True)
