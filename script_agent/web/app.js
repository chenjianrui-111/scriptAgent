const state = {
  streamReader: null,
};

function $(id) {
  return document.getElementById(id);
}

function nowTime() {
  return new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

function logLine(message) {
  const logs = $("logs");
  const li = document.createElement("li");
  li.innerHTML = `<time>${nowTime()}</time>${message}`;
  logs.prepend(li);
}

function setStatus(text, mode) {
  const badge = $("statusBadge");
  badge.textContent = text;
  if (mode === "ok") {
    badge.style.background = "#0f7b6c";
  } else if (mode === "err") {
    badge.style.background = "#be3f3f";
  } else if (mode === "run") {
    badge.style.background = "#d79100";
  } else {
    badge.style.background = "#445954";
  }
}

function collectHeaders(withJson = true) {
  const headers = {};
  if (withJson) {
    headers["Content-Type"] = "application/json";
  }

  const tenantId = $("tenantId").value.trim();
  const role = $("role").value.trim();
  const apiKey = $("apiKey").value.trim();
  const bearer = $("bearer").value.trim();

  if (tenantId) headers["X-Tenant-Id"] = tenantId;
  if (role) headers["X-Role"] = role;
  if (apiKey) headers["X-API-Key"] = apiKey;
  if (bearer) headers["Authorization"] = `Bearer ${bearer}`;

  return headers;
}

function buildUrl(path) {
  const base = $("baseUrl").value.trim().replace(/\/$/, "");
  if (!base) return path;
  return `${base}${path}`;
}

async function callApi(path, options = {}) {
  const response = await fetch(buildUrl(path), {
    ...options,
    headers: {
      ...collectHeaders(options.body !== undefined),
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }
  return response;
}

async function handleCreateSession() {
  try {
    setStatus("创建中", "run");
    const payload = {
      influencer_id: $("influencerId").value.trim(),
      influencer_name: $("influencerName").value.trim(),
      category: $("category").value.trim(),
    };

    const res = await callApi("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    $("sessionId").value = data.session_id;
    logLine(`会话已创建: <code>${data.session_id}</code>`);
    setStatus("会话就绪", "ok");
  } catch (err) {
    setStatus("创建失败", "err");
    logLine(`创建会话失败: ${err.message}`);
  }
}

function renderMeta(data) {
  const timing = data.timing_ms || {};
  const detail = [
    `intent: ${data.intent || "-"}`,
    `confidence: ${data.confidence ?? "-"}`,
    `quality_score: ${data.quality_score ?? "-"}`,
    `timing: ${JSON.stringify(timing)}`,
  ];
  $("meta").textContent = detail.join(" | ");
}

async function handleGenerateSync() {
  const sessionId = $("sessionId").value.trim();
  const query = $("query").value.trim();

  if (!sessionId || !query) {
    logLine("请先提供会话 ID 和用户请求");
    return;
  }

  try {
    setStatus("生成中", "run");
    $("output").textContent = "";
    const res = await callApi("/api/v1/generate", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, query }),
    });

    const data = await res.json();
    if (data.success) {
      $("output").textContent = data.script_content || "";
      setStatus("生成成功", "ok");
      logLine(`同步生成成功, trace_id=<code>${data.trace_id || "-"}</code>`);
    } else if (data.clarification_needed) {
      $("output").textContent = data.clarification_question || "需要更多信息";
      setStatus("需要澄清", "run");
      logLine("模型需要更多信息进行澄清");
    } else {
      $("output").textContent = data.error || "生成失败";
      setStatus("生成失败", "err");
      logLine(`生成失败: ${data.error || "unknown"}`);
    }
    renderMeta(data);
  } catch (err) {
    setStatus("生成失败", "err");
    $("output").textContent = err.message;
    logLine(`同步生成异常: ${err.message}`);
  }
}

async function handleGenerateStream() {
  const sessionId = $("sessionId").value.trim();
  const query = $("query").value.trim();

  if (!sessionId || !query) {
    logLine("请先提供会话 ID 和用户请求");
    return;
  }

  try {
    setStatus("流式生成中", "run");
    $("output").textContent = "";
    $("meta").textContent = "";

    const res = await callApi("/api/v1/generate/stream", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, query }),
    });

    if (!res.body) {
      throw new Error("浏览器不支持可读流");
    }

    const reader = res.body.getReader();
    state.streamReader = reader;
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      while (true) {
        const splitAt = buffer.indexOf("\n\n");
        if (splitAt < 0) break;

        const eventRaw = buffer.slice(0, splitAt);
        buffer = buffer.slice(splitAt + 2);

        for (const line of eventRaw.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const token = line.slice(6);

          if (token === "[DONE]") {
            setStatus("流式完成", "ok");
            logLine("流式生成完成");
            return;
          }
          if (token.startsWith("[ERROR]")) {
            setStatus("流式失败", "err");
            logLine(`流式异常: ${token}`);
            return;
          }

          $("output").textContent += token;
        }
      }
    }

    setStatus("流式结束", "ok");
  } catch (err) {
    setStatus("流式失败", "err");
    logLine(`流式调用异常: ${err.message}`);
  } finally {
    state.streamReader = null;
  }
}

async function handleStopStream() {
  if (!state.streamReader) {
    logLine("当前没有正在进行的流式任务");
    return;
  }
  try {
    await state.streamReader.cancel();
    setStatus("流式已停止", "err");
    logLine("已手动停止流式生成");
  } catch (err) {
    logLine(`停止流式失败: ${err.message}`);
  } finally {
    state.streamReader = null;
  }
}

function bindEvents() {
  $("createSessionBtn").addEventListener("click", handleCreateSession);
  $("generateBtn").addEventListener("click", handleGenerateSync);
  $("streamBtn").addEventListener("click", handleGenerateStream);
  $("stopStreamBtn").addEventListener("click", handleStopStream);

  $("query").addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      handleGenerateSync();
    }
  });
}

bindEvents();
logLine("前端已就绪，先创建会话再发起生成。");
