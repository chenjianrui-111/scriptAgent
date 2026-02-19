# ScriptAgent 监控面板

本项目已内置 Prometheus 指标出口 `GET /metrics`，并提供可直接启动的 Prometheus + Grafana 面板。

## 1. 监控范围

- 过程监控（Workflow 过程）
  - 阶段耗时 P95：`context_loading / intent_recognition / script_generation / quality_check / total`
  - Skill 命中分布
  - Checkpoint 写入状态
  - 锁超时速率

- 总体指标（服务健康）
  - API QPS
  - API P95 延迟
  - API 5xx 比例
  - 工作流成功率
  - 质量分中位数、意图置信度中位数
  - 活跃会话（进程内）
  - HTTP Inflight

## 2. 本地启动监控栈

在项目根目录执行（与业务服务同一个 compose project）：

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml up -d --build
```

访问地址：

- ScriptAgent API: `http://127.0.0.1:8080`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

Grafana 默认账号密码（可在 `.env` 覆盖）：

- 用户名：`admin`
- 密码：`admin123`

可覆盖变量：

```bash
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change-me
```

## 3. 验证指标是否生效

```bash
curl -sS http://127.0.0.1:8080/metrics | rg "script_agent_http_requests_total|script_agent_request_duration_seconds|script_agent_quality_score"
```

然后发起几次生成请求，再刷新 Grafana 面板（`ScriptAgent / ScriptAgent - 运行监控总览`）。

## 4. 线上部署（ECS）

在服务器项目目录执行：

```bash
cd /opt/script-agent
docker compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml up -d --build
```

若你只开放 Nginx 80/443，可将 3000/9090 仅对内网开放，Grafana 通过 Nginx 反代并加鉴权。

## 5. 安全建议

- 生产环境不要暴露 `9090` 到公网。
- Grafana 强制修改默认密码。
- 若需要多租户监控，建议接入 Loki/Tempo 并按 `tenant_id` 做标签化与权限控制。
