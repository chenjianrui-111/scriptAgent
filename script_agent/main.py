"""
话术助手 Agent 系统 - 启动入口

启动方式:
  开发环境: python -m script_agent.main
  生产环境: uvicorn script_agent.api.app:app --host 0.0.0.0 --port 8080 --workers 4
"""

import logging
import uvicorn

from script_agent.config.settings import settings

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    uvicorn.run(
        "script_agent.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
