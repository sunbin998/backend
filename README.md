# Backend Migration Guide

## Why

当前项目即将进入「用户登录 + 鉴权 + 数据隔离」阶段，推荐使用可回滚迁移，而不是直接删库。

## Added in this round

- Alembic 迁移框架：`alembic.ini` + `alembic/`
- 首个迁移：`alembic/versions/20260405_0001_auth_prepare.py`
	- 新增 `users` 表
	- 给 `categories/sessions/documents/diary_entries` 增加 `user_id`
	- 历史数据回填到 `legacy_user`
	- 支持 `downgrade` 回滚
- 第二个迁移：`alembic/versions/20260405_0002_auth_enforce_constraints.py`
	- 把业务表 `user_id` 改为 `NOT NULL`
	- `categories` 改为 `(user_id, name)` 联合唯一
	- `diary_entries` 改为 `(user_id, date)` 联合唯一

## Recommended DB strategy

优先采用 **新库并行开发**：

1. 保留旧库（例如 `RAGDB`）
2. 新建迁移开发库（例如 `RAGDB_AUTH_DEV`）
3. 在新库执行迁移并联调鉴权
4. 验证稳定后再迁移正式库

## Environment

在 `backend/.env` 里至少设置：

- `DATABASE_URL=postgresql+asyncpg://...`
- `SQL_ECHO=false`
- `DB_RECREATE_ON_START=false`

> 注意：启用 Alembic 后，不建议在真实环境依赖 `drop_all/create_all` 进行结构变更。

## Common commands

```bash
cd /python/VibeMVP/Graduate-RAG/backend

# 升级到最新版本
./.venv/bin/alembic upgrade head

# 查看当前版本
./.venv/bin/alembic current

# 查看历史版本
./.venv/bin/alembic history

# 回滚一步
./.venv/bin/alembic downgrade -1
```

## Notes

- 现在业务层已经按 `user_id` 做强隔离；数据库层也已收口到用户维度约束。
- 如需继续演进，可新增迁移将业务高频查询补充更多复合索引（例如 `sessions(user_id, created_at)`）。

