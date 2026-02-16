# ScriptAgent 阿里云 ECS 部署手册

本文整理了本次完整部署流程，适用于在阿里云 ECS（Ubuntu 24.04）上部署 `script-agent`。

---

## 1. 目标环境

- 系统：Ubuntu 24.04 LTS
- 公网 IP：示例 `47.99.119.146`
- 部署目录：`/opt/script-agent`
- 启动方式：`docker compose -f docker-compose.prod.yml up -d --build`

---

## 2. 本地准备

在本地机器（Mac）确认：

1. 项目代码在本地目录（示例）  
   `/Users/chenjianrui/Downloads/scriptAgent`
2. 持有 ECS 实例对应私钥（示例）  
   `~/Downloads/scriptAgent.pem`

---

## 3. 本地上传代码到 ECS

> 这一步必须在本地终端执行，不要在服务器里执行。

```bash
cd /Users/chenjianrui/Downloads/scriptAgent
chmod 600 ~/Downloads/scriptAgent.pem

ssh -i ~/Downloads/scriptAgent.pem root@47.99.119.146 "mkdir -p /opt/script-agent"

rsync -az --delete \
  -e "ssh -i ~/Downloads/scriptAgent.pem" \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude ".claude" \
  --exclude "tests/__pycache__" \
  /Users/chenjianrui/Downloads/scriptAgent/ \
  root@47.99.119.146:/opt/script-agent/
```

---

## 4. 服务器安装 Docker

登录服务器后执行：

```bash
ssh -i ~/Downloads/scriptAgent.pem root@47.99.119.146
cd /opt/script-agent

apt update
apt install -y docker.io docker-compose-v2 curl rsync
systemctl enable --now docker
docker --version
docker compose version
```

> 如果 `docker compose` 子命令不可用，可安装 `docker-compose` 并替换命令为 `docker-compose ...`。

---

## 5. Docker Hub 超时时的处理（已验证可用）

如果出现 `registry-1.docker.io ... i/o timeout`：

### 5.1 配置镜像加速

```bash
mkdir -p /etc/docker
cat >/etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": ["https://p8wkeele.mirror.aliyuncs.com"]
}
EOF
systemctl daemon-reload
systemctl restart docker
```

验证：

```bash
docker info | sed -n '/Registry Mirrors/,+5p'
curl -I --max-time 10 https://p8wkeele.mirror.aliyuncs.com/v2/
```

### 5.2 若仍回退到 Docker Hub，则改 Dockerfile 基础镜像

```bash
cd /opt/script-agent
sed -i 's|^FROM python:3.11-slim|FROM m.daocloud.io/docker.io/library/python:3.11-slim|' Dockerfile
```

---

## 6. 配置生产环境变量

```bash
cd /opt/script-agent
cp -n .env.production.example .env
nano .env
```

至少修改：

- `ZHIPU_API_KEY=你的真实key`
- 若有自建主模型：`VLLM_BASE_URL=...`
- 保留生成约束：
  - `SCRIPT_MIN_CHARS=40`
  - `SCRIPT_PRIMARY_ATTEMPTS=2`

> 注意：不要把真实 `.env` 提交到 Git 仓库。

---

## 7. 启动服务

```bash
cd /opt/script-agent
COMPOSE_BAKE=false DOCKER_BUILDKIT=0 docker compose -f docker-compose.prod.yml up -d --build
```

---

## 8. 服务验收

### 8.1 容器与健康检查

```bash
docker compose -f /opt/script-agent/docker-compose.prod.yml ps
ss -lntp | grep 8080
curl -sS http://127.0.0.1:8080/api/v1/health
```

预期：

- `script-agent` 状态 `healthy`
- `0.0.0.0:8080` 有监听
- `health` 返回 JSON 且 `status=healthy`

### 8.2 公网可达性检查

```bash
curl -v --connect-timeout 5 http://47.99.119.146:8080/api/v1/health
```

如果公网访问失败，优先检查阿里云安全组和云防火墙（见下一节）。

---

## 9. 阿里云网络放通要求

在 ECS 安全组「入方向」至少放行：

- `TCP 22`（仅你的办公 IP）
- `TCP 8080`（测试阶段可先 `0.0.0.0/0`）

若启用了阿里云云防火墙，也要同步放行 `8080/tcp`。

---

## 10. 线上接口快速验收命令

```bash
# 首页
curl -sS http://47.99.119.146:8080/ | head -n 20

# 健康检查
curl -sS http://47.99.119.146:8080/api/v1/health

# 生产环境需要认证，以下示例使用 X-API-Key
curl -sS -X POST http://47.99.119.146:8080/api/v1/sessions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-dev-test-key-001" \
  -d '{"influencer_name":"线上验收达人","category":"美妆"}'
```

---

## 11. 常见报错与解决

### 11.1 `Permission denied (publickey)`

原因：私钥路径错误或不是实例绑定密钥。  
处理：

- 必须在本地执行 `ssh/rsync`
- 确认 `-i ~/Downloads/xxx.pem` 存在且 `chmod 600`
- 核对 ECS 绑定的密钥对是否对应这把私钥

### 11.2 `docker-ce` / `docker-ce-cli` 找不到

原因：Docker CE 源不可用。  
处理：直接使用 Ubuntu 官方包：

```bash
apt install -y docker.io docker-compose-v2
```

### 11.3 `registry-1.docker.io ... timeout`

原因：Docker Hub 网络不可达。  
处理：

1. 配置阿里云镜像加速  
2. 仍失败则改 `FROM` 为 `m.daocloud.io/docker.io/library/python:3.11-slim`

### 11.4 浏览器打不开 `http://<公网IP>:8080/`

若容器内健康检查 OK，但公网超时，说明是网络层拦截：  
优先检查安全组/云防火墙/公网带宽配置。

### 11.5 偶发 `会话不存在`

多 worker + `SESSION_STORE=memory` 可能产生跨进程会话不一致。  
建议：

- 单机先用 `workers=1`，或
- 将会话存储切换到 `sqlite/redis` 再开启多 worker

---

## 12. 维护命令

```bash
# 查看状态
docker compose -f /opt/script-agent/docker-compose.prod.yml ps

# 追踪日志
docker compose -f /opt/script-agent/docker-compose.prod.yml logs -f script-agent

# 重启
docker compose -f /opt/script-agent/docker-compose.prod.yml up -d --build

# 停止
docker compose -f /opt/script-agent/docker-compose.prod.yml down
```

---

## 13. 安全建议

1. 立即轮换任何在聊天/截图中暴露过的 API Key。  
2. 生产不要继续使用示例 `sk-dev-test-key-001`。  
3. 使用域名 + HTTPS（Nginx/Caddy）对外发布。  
4. 安全组最小化开放，仅对必要端口和来源开放。
