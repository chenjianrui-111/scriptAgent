/* =========================================================
   ScriptAgent - Chat-style Script Assistant
   ========================================================= */

// ── Constants ─────────────────────────────────────────────

const FLOW = {
  INIT:            'init',
  SELECT_SCENARIO: 'select_scenario',
  SELECT_CATEGORY: 'select_category',
  SELECT_TYPE:     'select_type',
  SELECT_PRODUCT:  'select_product',
  CUSTOM_PRODUCT:  'custom_product',
  GENERATING:      'generating',
  CHAT:            'chat',
  ERROR:           'error',
};

// 后端支持的品类 (对齐 SlotExtractor.CATEGORY_KEYWORDS)
const CATEGORIES = [
  { id: 'beauty',   label: '\ud83d\udc84 美妆', value: '美妆' },
  { id: 'food',     label: '\ud83c\udf5c 食品', value: '食品' },
  { id: 'fashion',  label: '\ud83d\udc57 服饰', value: '服饰' },
  { id: 'digital',  label: '\ud83d\udcf1 数码', value: '数码' },
];

// 后端支持的话术类型 (对齐 SCENARIO_TEMPLATES)
const SCRIPT_TYPES = {
  live: [
    { label: '\ud83d\udc4b 开场话术', value: 'opening',        needProduct: false },
    { label: '\ud83d\udecd\ufe0f 卖点介绍', value: 'selling_points', needProduct: true },
    { label: '\ud83c\udf89 促销话术', value: 'promotion',      needProduct: true },
  ],
  short_video: [
    { label: '\ud83d\udc4b 开场白', value: 'opening',        needProduct: false },
    { label: '\ud83d\udecd\ufe0f 卖点介绍', value: 'selling_points', needProduct: true },
  ],
  seeding: [
    { label: '\ud83c\udf31 种草文案', value: 'seeding', needProduct: true },
  ],
};

// 类型到后端 sub_scenario 关键词映射
const TYPE_LABELS = {
  opening:        '开场白',
  selling_points: '卖点介绍',
  promotion:      '促销话术',
  seeding:        '种草文案',
};

const SCENARIO_LABELS = {
  live:        '直播',
  short_video: '短视频',
  seeding:     '种草',
};

const PRESET_PRODUCTS = [
  {
    id: 'beauty-lipstick', name: '丝绒口红套装', brand: '完美日记',
    category: '美妆', emoji: '\ud83d\udc84',
    features: ['丝绒哑光质地', '持久不脱色', '滋润不拔干'],
  },
  {
    id: 'beauty-serum', name: '玻尿酸精华液', brand: '薏诺娜',
    category: '美妆', emoji: '\u2728',
    features: ['高浓度玻尿酸', '深层补水', '敏感肌适用'],
  },
  {
    id: 'beauty-cushion', name: '气垫BB霜', brand: '花西子',
    category: '美妆', emoji: '\ud83e\ude9e',
    features: ['轻薄服帖', '养肤成分', '持妆12小时'],
  },
  {
    id: 'food-nuts', name: '每日坚果礼盒', brand: '三只松鼠',
    category: '食品', emoji: '\ud83e\udd5c',
    features: ['6种坚果混合', '锁鲜小包装', '零添加'],
  },
  {
    id: 'food-snack', name: '低卡魔芋爽', brand: '卫龙',
    category: '食品', emoji: '\ud83c\udf5c',
    features: ['低卡零食', '多种口味', '解馋不怕胖'],
  },
  {
    id: 'fashion-dress', name: '法式碎花连衣裙', brand: 'UR',
    category: '服饰', emoji: '\ud83d\udc57',
    features: ['法式浪漫风', '显瘦A字版型', '透气面料'],
  },
  {
    id: 'fashion-tshirt', name: '重磅纯棉T恤', brand: 'Bosie',
    category: '服饰', emoji: '\ud83d\udc55',
    features: ['260g重磅棉', '宽松廓形', '不变形不缩水'],
  },
  {
    id: 'digital-earbuds', name: '降噪蓝牙耳机', brand: '漫步者',
    category: '数码', emoji: '\ud83c\udfa7',
    features: ['主动降噪', '30小时续航', 'IP67防水'],
  },
  {
    id: 'digital-charger', name: '氮化镓快充头', brand: '安克',
    category: '数码', emoji: '\ud83d\udd0b',
    features: ['65W快充', '氮化镓小巧', '多协议兼容'],
  },
];

const BOT_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>';
const CLIENT_CONFIG_KEY = 'script_agent_frontend_config_v1';

const COPY_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
const REGEN_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>';
const EXPORT_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';

const STEPPER_STEPS = ['scenario', 'category', 'type', 'product', 'generate'];

const CATEGORY_DOT_MAP = {
  '美妆': 'beauty',
  '食品': 'food',
  '服饰': 'fashion',
  '数码': 'digital',
};

// ── State ─────────────────────────────────────────────────

const appState = {
  flowState: FLOW.INIT,
  sessionId: null,
  scenario: null,
  category: null,
  scriptType: null,
  product: null,
  streamReader: null,
  isStreaming: false,
  lastQuery: null,
  genStartTime: null,
};

// ── DOM refs ──────────────────────────────────────────────

const chatArea         = document.getElementById('chatArea');
const quickReplies     = document.getElementById('quickReplies');
const textInputWrap    = document.getElementById('textInputWrap');
const userInput        = document.getElementById('userInput');
const sendBtn          = document.getElementById('sendBtn');
const newSessionBtn    = document.getElementById('newSessionBtn');
const connToggleBtn    = document.getElementById('connToggleBtn');
const connPanel        = document.getElementById('connPanel');
const apiBaseUrl       = document.getElementById('apiBaseUrl');
const tenantIdInput    = document.getElementById('tenantIdInput');
const roleInput        = document.getElementById('roleInput');
const apiKeyInput      = document.getElementById('apiKeyInput');
const bearerTokenInput = document.getElementById('bearerTokenInput');
const saveConnBtn      = document.getElementById('saveConnBtn');
const sidebar          = document.getElementById('sidebar');
const sessionList      = document.getElementById('sessionList');
const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
const sidebarOpenBtn   = document.getElementById('sidebarOpenBtn');
const flowStepper      = document.getElementById('flowStepper');
const statusDot        = document.getElementById('statusDot');
const statsBar         = document.getElementById('statsBar');
const toastContainer   = document.getElementById('toastContainer');

// ── Helpers ───────────────────────────────────────────────

function escapeHtml(str) {
  const el = document.createElement('span');
  el.textContent = str;
  return el.innerHTML;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
  });
}

function hideAllInputs() {
  quickReplies.classList.add('hidden');
  quickReplies.innerHTML = '';
  textInputWrap.classList.add('hidden');
}

function loadClientConfig() {
  try {
    const raw = localStorage.getItem(CLIENT_CONFIG_KEY);
    if (!raw) return;
    const cfg = JSON.parse(raw);
    apiBaseUrl.value = cfg.baseUrl || '';
    tenantIdInput.value = cfg.tenantId || 'tenant_dev';
    roleInput.value = cfg.role || 'admin';
    apiKeyInput.value = cfg.apiKey || '';
    bearerTokenInput.value = cfg.bearerToken || '';
  } catch (_) {
    // ignore invalid cache
  }
}

function saveClientConfig() {
  const cfg = {
    baseUrl: (apiBaseUrl.value || '').trim(),
    tenantId: (tenantIdInput.value || '').trim() || 'tenant_dev',
    role: (roleInput.value || '').trim() || 'admin',
    apiKey: (apiKeyInput.value || '').trim(),
    bearerToken: (bearerTokenInput.value || '').trim(),
  };
  localStorage.setItem(CLIENT_CONFIG_KEY, JSON.stringify(cfg));
}

// ── Toast Notification System ─────────────────────────────

function showToast(message, type, durationMs) {
  type = type || 'info';
  durationMs = durationMs || 3000;

  var icons = { success: '\u2705', error: '\u274c', info: '\u2139\ufe0f' };

  var toast = document.createElement('div');
  toast.className = 'toast toast-' + type;
  toast.innerHTML =
    '<span class="toast-icon">' + (icons[type] || icons.info) + '</span>' +
    '<span>' + escapeHtml(message) + '</span>';

  toastContainer.appendChild(toast);

  setTimeout(function() {
    toast.classList.add('toast-exit');
    setTimeout(function() {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 300);
  }, durationMs);
}

// ── Status Indicator ──────────────────────────────────────

function setStatus(state) {
  statusDot.className = 'status-dot status-' + state;
  var titles = {
    idle: '就绪',
    connected: '已连接',
    generating: '生成中...',
    error: '出错',
  };
  statusDot.title = titles[state] || state;
}

// ── Flow Stepper ──────────────────────────────────────────

function updateStepper(currentStep) {
  var stepIndex = STEPPER_STEPS.indexOf(currentStep);
  if (stepIndex < 0) {
    flowStepper.classList.add('hidden');
    return;
  }

  flowStepper.classList.remove('hidden');
  var steps = flowStepper.querySelectorAll('.step');
  for (var i = 0; i < steps.length; i++) {
    steps[i].classList.remove('step-completed', 'step-active');
    if (i < stepIndex) {
      steps[i].classList.add('step-completed');
    } else if (i === stepIndex) {
      steps[i].classList.add('step-active');
    }
  }
}

// ── Grid Canvas Background ────────────────────────────────

function initGridCanvas() {
  var canvas = document.getElementById('gridCanvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var dpr = window.devicePixelRatio || 1;
  var spacing = 40;
  var dotSize = 1;
  var offset = 0;
  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function resize() {
    canvas.width = window.innerWidth * dpr;
    canvas.height = window.innerHeight * dpr;
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';
    ctx.scale(dpr, dpr);
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);
    ctx.fillStyle = 'rgba(0, 229, 255, 0.6)';

    var w = canvas.width / dpr;
    var h = canvas.height / dpr;
    var drift = reducedMotion ? 0 : Math.sin(offset * 0.005) * 5;

    for (var x = drift; x < w; x += spacing) {
      for (var y = drift; y < h; y += spacing) {
        ctx.beginPath();
        ctx.arc(x, y, dotSize, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (!reducedMotion) {
      offset++;
      requestAnimationFrame(draw);
    }
  }

  resize();
  window.addEventListener('resize', resize);
  draw();
}

// ── Typing Indicator ──────────────────────────────────────

function showTypingIndicator() {
  hideTypingIndicator();
  var el = document.createElement('div');
  el.className = 'msg msg-assistant';
  el.id = 'typing-indicator';
  el.innerHTML = '<div class="msg-avatar">' + BOT_SVG + '</div>' +
    '<div class="msg-body"><div class="typing-indicator">' +
    '<span></span><span></span><span></span></div></div>';
  chatArea.appendChild(el);
  scrollToBottom();
}

function hideTypingIndicator() {
  var el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

// ── Stats Bar ─────────────────────────────────────────────

function showStats(generationTimeMs, tokenCount) {
  document.getElementById('statTime').textContent = '\u23f1 ' + (generationTimeMs / 1000).toFixed(1) + 's';
  document.getElementById('statTokens').textContent = '\ud83d\udcdd ~' + tokenCount + ' tokens';
  document.getElementById('statQuality').textContent = '';
  statsBar.classList.remove('hidden');
}

function hideStats() {
  statsBar.classList.add('hidden');
}

// ── Script Action Buttons ─────────────────────────────────

function addScriptActions(msgId) {
  var msgEl = document.getElementById(msgId);
  if (!msgEl) return;

  var actionsDiv = document.createElement('div');
  actionsDiv.className = 'script-actions';

  var copyBtn = document.createElement('button');
  copyBtn.className = 'script-action-btn';
  copyBtn.innerHTML = COPY_SVG + '<span>复制</span>';
  copyBtn.addEventListener('click', function() { copyScript(msgId); });

  var regenBtn = document.createElement('button');
  regenBtn.className = 'script-action-btn';
  regenBtn.innerHTML = REGEN_SVG + '<span>重新生成</span>';
  regenBtn.addEventListener('click', function() { regenerateScript(); });

  var exportBtn = document.createElement('button');
  exportBtn.className = 'script-action-btn';
  exportBtn.innerHTML = EXPORT_SVG + '<span>导出</span>';
  exportBtn.addEventListener('click', function() { exportScript(msgId); });

  actionsDiv.appendChild(copyBtn);
  actionsDiv.appendChild(regenBtn);
  actionsDiv.appendChild(exportBtn);
  msgEl.querySelector('.msg-body').appendChild(actionsDiv);
}

function copyScript(msgId) {
  var el = document.getElementById(msgId);
  if (!el) return;
  var text = el.querySelector('.msg-content').textContent;
  try {
    navigator.clipboard.writeText(text).then(function() {
      showToast('已复制到剪贴板', 'success');
    });
  } catch (_) {
    // fallback
    var ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    showToast('已复制到剪贴板', 'success');
  }
}

function regenerateScript() {
  if (!appState.lastQuery || !appState.sessionId) return;
  appState.flowState = FLOW.GENERATING;
  doStreamGeneration(appState.lastQuery);
}

function exportScript(msgId) {
  var el = document.getElementById(msgId);
  if (!el) return;
  var text = el.querySelector('.msg-content').textContent;
  var blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'script-' + Date.now() + '.txt';
  a.click();
  URL.revokeObjectURL(a.href);
  showToast('已导出文件', 'success');
}

// ── Session Sidebar ───────────────────────────────────────

function toggleSidebar() {
  sidebar.classList.toggle('collapsed');
  sidebarOpenBtn.classList.toggle('hidden', !sidebar.classList.contains('collapsed'));
}

async function loadSessionList() {
  try {
    var res = await fetch(buildApiUrl('/api/v1/sessions'), {
      method: 'GET',
      headers: collectHeaders(),
    });
    if (!res.ok) {
      sessionList.innerHTML = '<div class="session-empty">无法加载历史</div>';
      return;
    }
    var sessions = await res.json();
    renderSessionList(sessions);
  } catch (_) {
    sessionList.innerHTML = '<div class="session-empty">无法加载历史</div>';
  }
}

function renderSessionList(sessions) {
  sessionList.innerHTML = '';
  if (!sessions || sessions.length === 0) {
    sessionList.innerHTML = '<div class="session-empty">暂无会话记录</div>';
    return;
  }

  for (var i = 0; i < sessions.length; i++) {
    var s = sessions[i];
    var item = document.createElement('div');
    item.className = 'session-item' + (s.session_id === appState.sessionId ? ' active' : '');
    item.dataset.sessionId = s.session_id;

    var dotClass = CATEGORY_DOT_MAP[s.category] || 'default';
    item.innerHTML =
      '<span class="session-dot session-dot-' + dotClass + '"></span>' +
      '<div class="session-info">' +
        '<div class="session-name">' + escapeHtml(s.influencer_name || s.session_id) + '</div>' +
        '<div class="session-meta">' + escapeHtml(s.category || '') + '</div>' +
      '</div>';

    item.addEventListener('click', (function(sid) {
      return function() { switchToSession(sid); };
    })(s.session_id));

    sessionList.appendChild(item);
  }
}

async function switchToSession(sessionId) {
  if (appState.isStreaming && appState.streamReader) {
    appState.streamReader.cancel();
    appState.streamReader = null;
    appState.isStreaming = false;
  }

  try {
    var res = await fetch(buildApiUrl('/api/v1/sessions/' + sessionId), {
      method: 'GET',
      headers: collectHeaders(),
    });
    if (!res.ok) {
      showToast('无法加载会话', 'error');
      return;
    }
    var detail = await res.json();

    chatArea.innerHTML = '';
    hideAllInputs();
    hideStats();

    appState.sessionId = sessionId;
    appState.flowState = FLOW.CHAT;

    // Render turn history
    if (detail.turns && detail.turns.length > 0) {
      for (var i = 0; i < detail.turns.length; i++) {
        var turn = detail.turns[i];
        if (turn.user) addUserMessage(turn.user);
        if (turn.assistant) addAssistantMessage(turn.assistant);
      }
    }

    showTextInput('继续对话...');
    setStatus('connected');
    updateStepper('generate');

    // Update active state in sidebar
    var items = sessionList.querySelectorAll('.session-item');
    for (var j = 0; j < items.length; j++) {
      items[j].classList.toggle('active', items[j].dataset.sessionId === sessionId);
    }

  } catch (err) {
    showToast('切换会话失败: ' + err.message, 'error');
  }
}

// ── Message Rendering ─────────────────────────────────────

function addAssistantMessage(text) {
  var id = 'msg-' + Date.now() + Math.random().toString(36).slice(2, 6);
  var el = document.createElement('div');
  el.className = 'msg msg-assistant';
  el.id = id;
  el.innerHTML =
    '<div class="msg-avatar">' + BOT_SVG + '</div>' +
    '<div class="msg-body"><div class="msg-content">' +
    escapeHtml(text).replace(/\n/g, '<br>') +
    '</div></div>';
  chatArea.appendChild(el);
  scrollToBottom();
  return id;
}

function addUserMessage(text) {
  var el = document.createElement('div');
  el.className = 'msg msg-user';
  el.innerHTML =
    '<div class="msg-body"><div class="msg-content">' +
    escapeHtml(text) +
    '</div></div>';
  chatArea.appendChild(el);
  scrollToBottom();
}

function addStreamingMessage() {
  var id = 'msg-' + Date.now() + Math.random().toString(36).slice(2, 6);
  var el = document.createElement('div');
  el.className = 'msg msg-assistant streaming';
  el.id = id;
  el.innerHTML =
    '<div class="msg-avatar">' + BOT_SVG + '</div>' +
    '<div class="msg-body"><div class="msg-content"></div></div>';
  chatArea.appendChild(el);
  scrollToBottom();
  appState.isStreaming = true;
  return id;
}

function updateStreamingMessage(msgId, token) {
  var el = document.getElementById(msgId);
  if (!el) return;
  el.querySelector('.msg-content').textContent += token;
  scrollToBottom();
}

function finalizeStreamingMessage(msgId) {
  var el = document.getElementById(msgId);
  if (!el) return;
  el.classList.remove('streaming');
}

// ── Input Controls ────────────────────────────────────────

function showQuickReplies(options) {
  hideAllInputs();
  quickReplies.classList.remove('hidden');
  for (var i = 0; i < options.length; i++) {
    var opt = options[i];
    var btn = document.createElement('button');
    btn.className = 'chip';
    btn.textContent = opt.label;
    btn.addEventListener('click', (function(val) {
      return function() {
        hideAllInputs();
        advanceFlow(val);
      };
    })(opt.value));
    quickReplies.appendChild(btn);
  }
}

function showProductCards() {
  hideAllInputs();
  quickReplies.classList.remove('hidden');

  var grid = document.createElement('div');
  grid.className = 'product-grid';

  // 按已选品类过滤商品
  var filtered = PRESET_PRODUCTS.filter(function(p) {
    return p.category === appState.category;
  });

  for (var i = 0; i < filtered.length; i++) {
    var p = filtered[i];
    var card = document.createElement('button');
    card.className = 'product-card';

    var tagsHtml = '';
    if (p.features && p.features.length > 0) {
      tagsHtml = '<div class="product-tags">';
      for (var j = 0; j < p.features.length; j++) {
        tagsHtml += '<span class="product-tag">' + escapeHtml(p.features[j]) + '</span>';
      }
      tagsHtml += '</div>';
    }

    card.innerHTML =
      '<span class="product-emoji">' + p.emoji + '</span>' +
      '<span class="product-name">' + escapeHtml(p.name) + '</span>' +
      '<span class="product-brand">' + escapeHtml(p.brand) + '</span>' +
      tagsHtml;
    card.addEventListener('click', (function(prod) {
      return function() {
        hideAllInputs();
        advanceFlow(prod);
      };
    })(p));
    grid.appendChild(card);
  }

  var customCard = document.createElement('button');
  customCard.className = 'product-card product-card-custom';
  customCard.innerHTML =
    '<span class="product-emoji">\u270f\ufe0f</span>' +
    '<span class="product-name">自定义商品</span>' +
    '<span class="product-brand">输入商品名称</span>';
  customCard.addEventListener('click', function() {
    hideAllInputs();
    advanceFlow('custom');
  });
  grid.appendChild(customCard);

  quickReplies.appendChild(grid);
}

function showTextInput(placeholder) {
  hideAllInputs();
  textInputWrap.classList.remove('hidden');
  userInput.value = '';
  userInput.placeholder = placeholder || '继续对话...';
  userInput.style.height = 'auto';
  sendBtn.disabled = true;
  userInput.focus();
}

// ── API Layer ─────────────────────────────────────────────

function buildApiUrl(path) {
  var base = (apiBaseUrl && apiBaseUrl.value ? apiBaseUrl.value : '').trim();
  if (!base) return path;
  return base.replace(/\/+$/, '') + path;
}

function collectHeaders() {
  var headers = {
    'Content-Type': 'application/json',
  };
  var tenantId = (tenantIdInput && tenantIdInput.value ? tenantIdInput.value : '').trim();
  var role = (roleInput && roleInput.value ? roleInput.value : '').trim();
  var apiKey = (apiKeyInput && apiKeyInput.value ? apiKeyInput.value : '').trim();
  var bearerToken = (bearerTokenInput && bearerTokenInput.value ? bearerTokenInput.value : '').trim();

  if (tenantId) headers['X-Tenant-Id'] = tenantId;
  if (role) headers['X-Role'] = role;
  if (apiKey) headers['X-API-Key'] = apiKey;
  if (bearerToken) headers.Authorization = 'Bearer ' + bearerToken;
  return headers;
}

async function createSession(influencerName, category) {
  var res = await fetch(buildApiUrl('/api/v1/sessions'), {
    method: 'POST',
    headers: collectHeaders(),
    body: JSON.stringify({
      influencer_name: influencerName,
      category: category,
    }),
  });
  if (!res.ok) throw new Error('Session creation failed: ' + res.status);
  var data = await res.json();
  return data.session_id;
}

async function generateStream(sessionId, query, onToken, onDone, onError) {
  var res = await fetch(buildApiUrl('/api/v1/generate/stream'), {
    method: 'POST',
    headers: collectHeaders(),
    body: JSON.stringify({ session_id: sessionId, query: query }),
  });
  if (!res.ok) {
    var text = await res.text();
    throw new Error('Generate failed: ' + res.status + ' ' + text);
  }

  var reader = res.body.getReader();
  appState.streamReader = reader;
  var decoder = new TextDecoder();
  var buffer = '';

  try {
    while (true) {
      var result = await reader.read();
      if (result.done) break;
      buffer += decoder.decode(result.value, { stream: true });

      while (true) {
        var idx = buffer.indexOf('\n\n');
        if (idx < 0) break;
        var raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);

        var lines = raw.split('\n');
        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (!line.startsWith('data: ')) continue;
          var token = line.slice(6);
          if (token === '[DONE]') { onDone(); return; }
          if (token.startsWith('[ERROR]')) { onError(token); return; }
          onToken(token);
        }
      }
    }
    onDone();
  } finally {
    appState.streamReader = null;
  }
}

// ── Query Construction ────────────────────────────────────

function buildQuery() {
  var scenarioLabel = SCENARIO_LABELS[appState.scenario] || '直播';
  var typeLabel = TYPE_LABELS[appState.scriptType] || '开场白';
  var categoryLabel = appState.category || '美妆';

  var query = '请为我生成一段' + categoryLabel + scenarioLabel + typeLabel + '话术';

  if (appState.product) {
    query += '，商品：' + appState.product.name;
    if (appState.product.brand) {
      query += '，品牌：' + appState.product.brand;
    }
    if (appState.product.features && appState.product.features.length > 0) {
      query += '，卖点：' + appState.product.features.join('、');
    }
  }

  query += '，语气要热情有感染力。';
  return query;
}

// ── Streaming Generation ──────────────────────────────────

async function doStreamGeneration(query) {
  hideAllInputs();
  hideStats();
  setStatus('generating');
  appState.lastQuery = query;
  appState.genStartTime = Date.now();

  showTypingIndicator();

  // Brief delay so typing indicator is visible
  await new Promise(function(r) { setTimeout(r, 400); });

  hideTypingIndicator();
  var msgId = addStreamingMessage();
  var charCount = 0;

  try {
    await generateStream(
      appState.sessionId,
      query,
      function onToken(token) {
        updateStreamingMessage(msgId, token);
        charCount += token.length;
      },
      function onDone() {
        finalizeStreamingMessage(msgId);
        addScriptActions(msgId);
        appState.flowState = FLOW.CHAT;
        appState.isStreaming = false;
        setStatus('connected');
        var elapsed = Date.now() - appState.genStartTime;
        showStats(elapsed, Math.max(1, Math.round(charCount / 1.5)));
        showTextInput('继续对话，例如：换一个更活泼的风格...');
        loadSessionList();
      },
      function onError(errorMsg) {
        finalizeStreamingMessage(msgId);
        addAssistantMessage('生成出现问题：' + errorMsg);
        appState.flowState = FLOW.CHAT;
        appState.isStreaming = false;
        setStatus('error');
        showTextInput();
      }
    );
  } catch (err) {
    finalizeStreamingMessage(msgId);
    addAssistantMessage('出现错误：' + err.message);
    appState.flowState = FLOW.CHAT;
    appState.isStreaming = false;
    setStatus('error');
    showTextInput();
  }
}

// ── Flow Controller (State Machine) ──────────────────────

async function advanceFlow(input) {
  switch (appState.flowState) {

    case FLOW.INIT: {
      appState.flowState = FLOW.SELECT_SCENARIO;
      updateStepper('scenario');
      addAssistantMessage('你好！我是你的话术助手\n请选择你想要生成的场景：');
      showQuickReplies([
        { label: '\ud83c\udfa4 直播带货', value: 'live' },
        { label: '\ud83c\udfac 短视频',   value: 'short_video' },
        { label: '\ud83c\udf31 种草文案', value: 'seeding' },
      ]);
      break;
    }

    case FLOW.SELECT_SCENARIO: {
      appState.scenario = input;
      addUserMessage(SCENARIO_LABELS[input] || input);
      appState.flowState = FLOW.SELECT_CATEGORY;
      updateStepper('category');
      addAssistantMessage('请选择商品品类：');
      showQuickReplies(CATEGORIES.map(function(c) {
        return { label: c.label, value: c.value };
      }));
      break;
    }

    case FLOW.SELECT_CATEGORY: {
      appState.category = input;
      addUserMessage(input);
      updateStepper('type');

      // 创建 session（用实际品类而非硬编码）
      try {
        appState.sessionId = await createSession('小雅', appState.category);
        setStatus('connected');
      } catch (err) {
        addAssistantMessage('创建会话失败，请刷新重试。');
        appState.flowState = FLOW.ERROR;
        setStatus('error');
        break;
      }

      var types = SCRIPT_TYPES[appState.scenario] || SCRIPT_TYPES.live;

      // 种草文案只有一种类型，直接跳到商品选择
      if (types.length === 1 && types[0].needProduct) {
        appState.scriptType = types[0].value;
        addAssistantMessage('请选择要推荐的商品：');
        appState.flowState = FLOW.SELECT_PRODUCT;
        updateStepper('product');
        showProductCards();
      } else {
        appState.flowState = FLOW.SELECT_TYPE;
        addAssistantMessage('想要生成哪种类型的话术？');
        showQuickReplies(types.map(function(t) {
          return { label: t.label, value: t.value };
        }));
      }
      break;
    }

    case FLOW.SELECT_TYPE: {
      appState.scriptType = input;
      addUserMessage(TYPE_LABELS[input] || input);

      // 查找当前类型是否需要商品
      var currentTypes = SCRIPT_TYPES[appState.scenario] || SCRIPT_TYPES.live;
      var typeInfo = currentTypes.find(function(t) { return t.value === input; });
      var needProduct = typeInfo ? typeInfo.needProduct : false;

      if (needProduct) {
        appState.flowState = FLOW.SELECT_PRODUCT;
        updateStepper('product');
        addAssistantMessage('请选择要推荐的商品，或者输入自定义商品名称：');
        showProductCards();
      } else {
        appState.flowState = FLOW.GENERATING;
        updateStepper('generate');
        await doStreamGeneration(buildQuery());
      }
      break;
    }

    case FLOW.SELECT_PRODUCT: {
      if (input === 'custom') {
        appState.flowState = FLOW.CUSTOM_PRODUCT;
        addAssistantMessage('请输入商品名称：');
        showTextInput('输入商品名称...');
      } else {
        appState.product = input;
        addUserMessage(input.name);
        appState.flowState = FLOW.GENERATING;
        updateStepper('generate');
        await doStreamGeneration(buildQuery());
      }
      break;
    }

    case FLOW.CUSTOM_PRODUCT: {
      appState.product = { id: 'custom', name: input, brand: '', category: appState.category || '', features: [] };
      addUserMessage(input);
      appState.flowState = FLOW.GENERATING;
      updateStepper('generate');
      await doStreamGeneration(buildQuery());
      break;
    }

    case FLOW.CHAT: {
      addUserMessage(input);
      appState.flowState = FLOW.GENERATING;
      await doStreamGeneration(input);
      break;
    }
  }
}

// ── Event Handlers ────────────────────────────────────────

function handleSend() {
  var text = userInput.value.trim();
  if (!text || appState.isStreaming) return;
  userInput.value = '';
  userInput.style.height = 'auto';
  sendBtn.disabled = true;
  advanceFlow(text);
}

userInput.addEventListener('input', function() {
  // auto-grow
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
  // toggle send button
  sendBtn.disabled = !userInput.value.trim() || appState.isStreaming;
});

userInput.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

sendBtn.addEventListener('click', handleSend);

newSessionBtn.addEventListener('click', function() {
  if (appState.isStreaming && appState.streamReader) {
    appState.streamReader.cancel();
    appState.streamReader = null;
    appState.isStreaming = false;
  }
  init();
});

connToggleBtn.addEventListener('click', function() {
  connPanel.classList.toggle('hidden');
});

saveConnBtn.addEventListener('click', function() {
  saveClientConfig();
  showToast('连接配置已保存', 'success');
  connPanel.classList.add('hidden');
});

sidebarToggleBtn.addEventListener('click', function() {
  toggleSidebar();
});

sidebarOpenBtn.addEventListener('click', function() {
  toggleSidebar();
});

// ── Keyboard Shortcuts ────────────────────────────────────

document.addEventListener('keydown', function(e) {
  // Ctrl+Enter or Cmd+Enter: send message
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    handleSend();
  }
  // Escape: close connection panel
  if (e.key === 'Escape') {
    if (!connPanel.classList.contains('hidden')) {
      connPanel.classList.add('hidden');
    }
  }
});

// ── Initialization ────────────────────────────────────────

async function init() {
  loadClientConfig();
  chatArea.innerHTML = '';
  hideAllInputs();
  hideStats();

  appState.flowState = FLOW.INIT;
  appState.sessionId = null;
  appState.scenario = null;
  appState.category = null;
  appState.scriptType = null;
  appState.product = null;
  appState.lastQuery = null;

  setStatus('idle');
  updateStepper(null);
  initGridCanvas();
  loadSessionList();

  await advanceFlow();
}

document.addEventListener('DOMContentLoaded', init);
