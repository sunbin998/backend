"""prepare auth schema with users and user-scoped columns

Revision ID: 20260405_0001
Revises:
Create Date: 2026-04-05 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "20260405_0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


LEGACY_USER_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    # 1) users 表（先落地，兼容后续鉴权）
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("username", sa.String(length=50), nullable=False),
        sa.Column("email", sa.String(length=100), nullable=True),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("avatar", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # 2) 预置 legacy 用户用于平滑迁移历史数据
    op.execute(
        sa.text(
            """
            INSERT INTO users (id, username, email, hashed_password)
            VALUES (CAST(:uid AS uuid), :uname, :email, :pwd)
            """
        ).bindparams(
            uid=LEGACY_USER_ID,
            uname="legacy_user",
            email="legacy_user@local.invalid",
            pwd="!migrated-no-login!",
        )
    )

    # 3) 给现有业务表加 user_id（第一阶段保持 nullable，避免立刻破坏现有业务写入）
    op.add_column("categories", sa.Column("user_id", postgresql.UUID(as_uuid=False), nullable=True))
    op.add_column("sessions", sa.Column("user_id", postgresql.UUID(as_uuid=False), nullable=True))
    op.add_column("documents", sa.Column("user_id", postgresql.UUID(as_uuid=False), nullable=True))
    op.add_column("diary_entries", sa.Column("user_id", postgresql.UUID(as_uuid=False), nullable=True))

    # 4) 回填历史数据到 legacy_user
    op.execute(sa.text("UPDATE categories SET user_id = CAST(:uid AS uuid) WHERE user_id IS NULL").bindparams(uid=LEGACY_USER_ID))
    op.execute(sa.text("UPDATE sessions SET user_id = CAST(:uid AS uuid) WHERE user_id IS NULL").bindparams(uid=LEGACY_USER_ID))
    op.execute(sa.text("UPDATE documents SET user_id = CAST(:uid AS uuid) WHERE user_id IS NULL").bindparams(uid=LEGACY_USER_ID))
    op.execute(sa.text("UPDATE diary_entries SET user_id = CAST(:uid AS uuid) WHERE user_id IS NULL").bindparams(uid=LEGACY_USER_ID))

    # 5) 增加索引与外键（保持 user_id 可空，便于过渡）
    op.create_index("ix_categories_user_id", "categories", ["user_id"], unique=False)
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"], unique=False)
    op.create_index("ix_documents_user_id", "documents", ["user_id"], unique=False)
    op.create_index("ix_diary_entries_user_id", "diary_entries", ["user_id"], unique=False)

    op.create_foreign_key("fk_categories_user_id_users", "categories", "users", ["user_id"], ["id"], ondelete="CASCADE")
    op.create_foreign_key("fk_sessions_user_id_users", "sessions", "users", ["user_id"], ["id"], ondelete="CASCADE")
    op.create_foreign_key("fk_documents_user_id_users", "documents", "users", ["user_id"], ["id"], ondelete="CASCADE")
    op.create_foreign_key("fk_diary_entries_user_id_users", "diary_entries", "users", ["user_id"], ["id"], ondelete="CASCADE")


def downgrade() -> None:
    op.drop_constraint("fk_diary_entries_user_id_users", "diary_entries", type_="foreignkey")
    op.drop_constraint("fk_documents_user_id_users", "documents", type_="foreignkey")
    op.drop_constraint("fk_sessions_user_id_users", "sessions", type_="foreignkey")
    op.drop_constraint("fk_categories_user_id_users", "categories", type_="foreignkey")

    op.drop_index("ix_diary_entries_user_id", table_name="diary_entries")
    op.drop_index("ix_documents_user_id", table_name="documents")
    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_index("ix_categories_user_id", table_name="categories")

    op.drop_column("diary_entries", "user_id")
    op.drop_column("documents", "user_id")
    op.drop_column("sessions", "user_id")
    op.drop_column("categories", "user_id")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
