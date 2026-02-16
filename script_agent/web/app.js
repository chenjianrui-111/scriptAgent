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
  { id: 'beauty',   label: '\ud83d\udc84 \u7f8e\u5986', value: '\u7f8e\u5986' },
  { id: 'food',     label: '\ud83c\udf5c \u98df\u54c1', value: '\u98df\u54c1' },
  { id: 'fashion',  label: '\ud83d\udc57 \u670d\u9970', value: '\u670d\u9970' },
  { id: 'digital',  label: '\ud83d\udcf1 \u6570\u7801', value: '\u6570\u7801' },
];

// 后端支持的话术类型 (对齐 SCENARIO_TEMPLATES)
const SCRIPT_TYPES = {
  live: [
    { label: '\ud83d\udc4b \u5f00\u573a\u8bdd\u672f', value: 'opening',        needProduct: false },
    { label: '\ud83d\udecd\ufe0f \u5356\u70b9\u4ecb\u7ecd', value: 'selling_points', needProduct: true },
    { label: '\ud83c\udf89 \u4fc3\u9500\u8bdd\u672f', value: 'promotion',      needProduct: true },
  ],
  short_video: [
    { label: '\ud83d\udc4b \u5f00\u573a\u767d', value: 'opening',        needProduct: false },
    { label: '\ud83d\udecd\ufe0f \u5356\u70b9\u4ecb\u7ecd', value: 'selling_points', needProduct: true },
  ],
  seeding: [
    { label: '\ud83c\udf31 \u79cd\u8349\u6587\u6848', value: 'seeding', needProduct: true },
  ],
};

// 类型到后端 sub_scenario 关键词映射
const TYPE_LABELS = {
  opening:        '\u5f00\u573a\u767d',
  selling_points: '\u5356\u70b9\u4ecb\u7ecd',
  promotion:      '\u4fc3\u9500\u8bdd\u672f',
  seeding:        '\u79cd\u8349\u6587\u6848',
};

const SCENARIO_LABELS = {
  live:        '\u76f4\u64ad',
  short_video: '\u77ed\u89c6\u9891',
  seeding:     '\u79cd\u8349',
};

const PRESET_PRODUCTS = [
  {
    id: 'beauty-lipstick', name: '\u4e1d\u7ed2\u53e3\u7ea2\u5957\u88c5', brand: '\u5b8c\u7f8e\u65e5\u8bb0',
    category: '\u7f8e\u5986', emoji: '\ud83d\udc84',
    features: ['\u4e1d\u7ed2\u54d1\u5149\u8d28\u5730', '\u6301\u4e45\u4e0d\u8131\u8272', '\u6ecb\u6da6\u4e0d\u62d4\u5e72'],
  },
  {
    id: 'beauty-serum', name: '\u73bb\u5c3f\u9178\u7cbe\u534e\u6db2', brand: '\u858f\u8bfa\u5a1c',
    category: '\u7f8e\u5986', emoji: '\u2728',
    features: ['\u9ad8\u6d53\u5ea6\u73bb\u5c3f\u9178', '\u6df1\u5c42\u8865\u6c34', '\u654f\u611f\u808c\u9002\u7528'],
  },
  {
    id: 'beauty-cushion', name: '\u6c14\u57abBB\u971c', brand: '\u82b1\u897f\u5b50',
    category: '\u7f8e\u5986', emoji: '\ud83e\ude9e',
    features: ['\u8f7b\u8584\u670d\u5e16', '\u517b\u80a4\u6210\u5206', '\u6301\u598612\u5c0f\u65f6'],
  },
  {
    id: 'food-nuts', name: '\u6bcf\u65e5\u575a\u679c\u793c\u76d2', brand: '\u4e09\u53ea\u677e\u9f20',
    category: '\u98df\u54c1', emoji: '\ud83e\udd5c',
    features: ['6\u79cd\u575a\u679c\u6df7\u5408', '\u9501\u9c9c\u5c0f\u5305\u88c5', '\u96f6\u6dfb\u52a0'],
  },
  {
    id: 'food-snack', name: '\u4f4e\u5361\u9b54\u828b\u723d', brand: '\u536b\u9f99',
    category: '\u98df\u54c1', emoji: '\ud83c\udf5c',
    features: ['\u4f4e\u5361\u96f6\u98df', '\u591a\u79cd\u53e3\u5473', '\u89e3\u998b\u4e0d\u6015\u80d6'],
  },
  {
    id: 'fashion-dress', name: '\u6cd5\u5f0f\u788e\u82b1\u8fde\u8863\u88d9', brand: 'UR',
    category: '\u670d\u9970', emoji: '\ud83d\udc57',
    features: ['\u6cd5\u5f0f\u6d6a\u6f2b\u98ce', '\u663e\u7626A\u5b57\u7248\u578b', '\u900f\u6c14\u9762\u6599'],
  },
  {
    id: 'fashion-tshirt', name: '\u91cd\u78c5\u7eaf\u68c9T\u6064', brand: 'Bosie',
    category: '\u670d\u9970', emoji: '\ud83d\udc55',
    features: ['260g\u91cd\u78c5\u68c9', '\u5bbd\u677e\u5ed3\u5f62', '\u4e0d\u53d8\u5f62\u4e0d\u7f29\u6c34'],
  },
  {
    id: 'digital-earbuds', name: '\u964d\u566a\u84dd\u7259\u8033\u673a', brand: '\u6f2b\u6b65\u8005',
    category: '\u6570\u7801', emoji: '\ud83c\udfa7',
    features: ['\u4e3b\u52a8\u964d\u566a', '30\u5c0f\u65f6\u7eed\u822a', 'IP67\u9632\u6c34'],
  },
  {
    id: 'digital-charger', name: '\u6c2e\u5316\u956d\u5feb\u5145\u5934', brand: '\u5b89\u514b',
    category: '\u6570\u7801', emoji: '\ud83d\udd0b',
    features: ['65W\u5feb\u5145', '\u6c2e\u5316\u956d\u5c0f\u5de7', '\u591a\u534f\u8bae\u517c\u5bb9'],
  },
];

const BOT_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>';

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
};

// ── DOM refs ──────────────────────────────────────────────

const chatArea      = document.getElementById('chatArea');
const quickReplies  = document.getElementById('quickReplies');
const textInputWrap = document.getElementById('textInputWrap');
const userInput     = document.getElementById('userInput');
const sendBtn       = document.getElementById('sendBtn');
const newSessionBtn = document.getElementById('newSessionBtn');

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

// ── Message Rendering ─────────────────────────────────────

function addAssistantMessage(text) {
  const id = 'msg-' + Date.now() + Math.random().toString(36).slice(2, 6);
  const el = document.createElement('div');
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
  const el = document.createElement('div');
  el.className = 'msg msg-user';
  el.innerHTML =
    '<div class="msg-body"><div class="msg-content">' +
    escapeHtml(text) +
    '</div></div>';
  chatArea.appendChild(el);
  scrollToBottom();
}

function addStreamingMessage() {
  const id = 'msg-' + Date.now() + Math.random().toString(36).slice(2, 6);
  const el = document.createElement('div');
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
  const el = document.getElementById(msgId);
  if (!el) return;
  el.querySelector('.msg-content').textContent += token;
  scrollToBottom();
}

function finalizeStreamingMessage(msgId) {
  const el = document.getElementById(msgId);
  if (!el) return;
  el.classList.remove('streaming');
}

// ── Input Controls ────────────────────────────────────────

function showQuickReplies(options) {
  hideAllInputs();
  quickReplies.classList.remove('hidden');
  for (const opt of options) {
    const btn = document.createElement('button');
    btn.className = 'chip';
    btn.textContent = opt.label;
    btn.addEventListener('click', () => {
      hideAllInputs();
      advanceFlow(opt.value);
    });
    quickReplies.appendChild(btn);
  }
}

function showProductCards() {
  hideAllInputs();
  quickReplies.classList.remove('hidden');

  const grid = document.createElement('div');
  grid.className = 'product-grid';

  // 按已选品类过滤商品
  const filtered = PRESET_PRODUCTS.filter(
    p => p.category === appState.category
  );

  for (const p of filtered) {
    const card = document.createElement('button');
    card.className = 'product-card';
    card.innerHTML =
      '<span class="product-emoji">' + p.emoji + '</span>' +
      '<span class="product-name">' + escapeHtml(p.name) + '</span>' +
      '<span class="product-brand">' + escapeHtml(p.brand) + '</span>';
    card.addEventListener('click', () => {
      hideAllInputs();
      advanceFlow(p);
    });
    grid.appendChild(card);
  }

  const customCard = document.createElement('button');
  customCard.className = 'product-card product-card-custom';
  customCard.innerHTML =
    '<span class="product-emoji">\u270f\ufe0f</span>' +
    '<span class="product-name">\u81ea\u5b9a\u4e49\u5546\u54c1</span>' +
    '<span class="product-brand">\u8f93\u5165\u5546\u54c1\u540d\u79f0</span>';
  customCard.addEventListener('click', () => {
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
  userInput.placeholder = placeholder || '\u7ee7\u7eed\u5bf9\u8bdd...';
  userInput.style.height = 'auto';
  sendBtn.disabled = true;
  userInput.focus();
}

// ── API Layer ─────────────────────────────────────────────

function collectHeaders() {
  return {
    'Content-Type': 'application/json',
    'X-Tenant-Id': 'tenant_dev',
    'X-Role': 'admin',
  };
}

async function createSession(influencerName, category) {
  const res = await fetch('/api/v1/sessions', {
    method: 'POST',
    headers: collectHeaders(),
    body: JSON.stringify({
      influencer_name: influencerName,
      category: category,
    }),
  });
  if (!res.ok) throw new Error('Session creation failed: ' + res.status);
  const data = await res.json();
  return data.session_id;
}

async function generateStream(sessionId, query, onToken, onDone, onError) {
  const res = await fetch('/api/v1/generate/stream', {
    method: 'POST',
    headers: collectHeaders(),
    body: JSON.stringify({ session_id: sessionId, query: query }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error('Generate failed: ' + res.status + ' ' + text);
  }

  const reader = res.body.getReader();
  appState.streamReader = reader;
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      while (true) {
        const idx = buffer.indexOf('\n\n');
        if (idx < 0) break;
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);

        for (const line of raw.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const token = line.slice(6);
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
  var scenarioLabel = SCENARIO_LABELS[appState.scenario] || '\u76f4\u64ad';
  var typeLabel = TYPE_LABELS[appState.scriptType] || '\u5f00\u573a\u767d';
  var categoryLabel = appState.category || '\u7f8e\u5986';

  var query = '\u8bf7\u4e3a\u6211\u751f\u6210\u4e00\u6bb5' + categoryLabel + scenarioLabel + typeLabel + '\u8bdd\u672f';

  if (appState.product) {
    query += '\uff0c\u5546\u54c1\uff1a' + appState.product.name;
    if (appState.product.brand) {
      query += '\uff0c\u54c1\u724c\uff1a' + appState.product.brand;
    }
    if (appState.product.features && appState.product.features.length > 0) {
      query += '\uff0c\u5356\u70b9\uff1a' + appState.product.features.join('\u3001');
    }
  }

  query += '\uff0c\u8bed\u6c14\u8981\u70ed\u60c5\u6709\u611f\u67d3\u529b\u3002';
  return query;
}

// ── Streaming Generation ──────────────────────────────────

async function doStreamGeneration(query) {
  hideAllInputs();
  const msgId = addStreamingMessage();

  try {
    await generateStream(
      appState.sessionId,
      query,
      function onToken(token) {
        updateStreamingMessage(msgId, token);
      },
      function onDone() {
        finalizeStreamingMessage(msgId);
        appState.flowState = FLOW.CHAT;
        appState.isStreaming = false;
        showTextInput('\u7ee7\u7eed\u5bf9\u8bdd\uff0c\u4f8b\u5982\uff1a\u6362\u4e00\u4e2a\u66f4\u6d3b\u6cfc\u7684\u98ce\u683c...');
      },
      function onError(errorMsg) {
        finalizeStreamingMessage(msgId);
        addAssistantMessage('\u751f\u6210\u51fa\u73b0\u95ee\u9898\uff0c\u8bf7\u91cd\u8bd5\u3002');
        appState.flowState = FLOW.CHAT;
        appState.isStreaming = false;
        showTextInput();
      }
    );
  } catch (err) {
    finalizeStreamingMessage(msgId);
    addAssistantMessage('\u51fa\u73b0\u9519\u8bef\uff1a' + err.message);
    appState.flowState = FLOW.CHAT;
    appState.isStreaming = false;
    showTextInput();
  }
}

// ── Flow Controller (State Machine) ──────────────────────

async function advanceFlow(input) {
  switch (appState.flowState) {

    case FLOW.INIT: {
      appState.flowState = FLOW.SELECT_SCENARIO;
      addAssistantMessage('\u4f60\u597d\uff01\u6211\u662f\u4f60\u7684\u8bdd\u672f\u52a9\u624b\n\u8bf7\u9009\u62e9\u4f60\u60f3\u8981\u751f\u6210\u7684\u573a\u666f\uff1a');
      showQuickReplies([
        { label: '\ud83c\udfa4 \u76f4\u64ad\u5e26\u8d27', value: 'live' },
        { label: '\ud83c\udfac \u77ed\u89c6\u9891',   value: 'short_video' },
        { label: '\ud83c\udf31 \u79cd\u8349\u6587\u6848', value: 'seeding' },
      ]);
      break;
    }

    case FLOW.SELECT_SCENARIO: {
      appState.scenario = input;
      addUserMessage(SCENARIO_LABELS[input] || input);
      appState.flowState = FLOW.SELECT_CATEGORY;
      addAssistantMessage('\u8bf7\u9009\u62e9\u5546\u54c1\u54c1\u7c7b\uff1a');
      showQuickReplies(CATEGORIES.map(function(c) {
        return { label: c.label, value: c.value };
      }));
      break;
    }

    case FLOW.SELECT_CATEGORY: {
      appState.category = input;
      addUserMessage(input);

      // 创建 session（用实际品类而非硬编码）
      try {
        appState.sessionId = await createSession('\u5c0f\u96c5', appState.category);
      } catch (err) {
        addAssistantMessage('\u521b\u5efa\u4f1a\u8bdd\u5931\u8d25\uff0c\u8bf7\u5237\u65b0\u91cd\u8bd5\u3002');
        appState.flowState = FLOW.ERROR;
        break;
      }

      var types = SCRIPT_TYPES[appState.scenario] || SCRIPT_TYPES.live;

      // 种草文案只有一种类型，直接跳到商品选择
      if (types.length === 1 && types[0].needProduct) {
        appState.scriptType = types[0].value;
        addAssistantMessage('\u8bf7\u9009\u62e9\u8981\u63a8\u8350\u7684\u5546\u54c1\uff1a');
        appState.flowState = FLOW.SELECT_PRODUCT;
        showProductCards();
      } else {
        appState.flowState = FLOW.SELECT_TYPE;
        addAssistantMessage('\u60f3\u8981\u751f\u6210\u54ea\u79cd\u7c7b\u578b\u7684\u8bdd\u672f\uff1f');
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
        addAssistantMessage('\u8bf7\u9009\u62e9\u8981\u63a8\u8350\u7684\u5546\u54c1\uff0c\u6216\u8005\u8f93\u5165\u81ea\u5b9a\u4e49\u5546\u54c1\u540d\u79f0\uff1a');
        showProductCards();
      } else {
        appState.flowState = FLOW.GENERATING;
        await doStreamGeneration(buildQuery());
      }
      break;
    }

    case FLOW.SELECT_PRODUCT: {
      if (input === 'custom') {
        appState.flowState = FLOW.CUSTOM_PRODUCT;
        addAssistantMessage('\u8bf7\u8f93\u5165\u5546\u54c1\u540d\u79f0\uff1a');
        showTextInput('\u8f93\u5165\u5546\u54c1\u540d\u79f0...');
      } else {
        appState.product = input;
        addUserMessage(input.name);
        appState.flowState = FLOW.GENERATING;
        await doStreamGeneration(buildQuery());
      }
      break;
    }

    case FLOW.CUSTOM_PRODUCT: {
      appState.product = { id: 'custom', name: input, brand: '', category: appState.category || '', features: [] };
      addUserMessage(input);
      appState.flowState = FLOW.GENERATING;
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
  const text = userInput.value.trim();
  if (!text || appState.isStreaming) return;
  userInput.value = '';
  userInput.style.height = 'auto';
  sendBtn.disabled = true;
  advanceFlow(text);
}

userInput.addEventListener('input', () => {
  // auto-grow
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
  // toggle send button
  sendBtn.disabled = !userInput.value.trim() || appState.isStreaming;
});

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

sendBtn.addEventListener('click', handleSend);

newSessionBtn.addEventListener('click', () => {
  if (appState.isStreaming && appState.streamReader) {
    appState.streamReader.cancel();
    appState.streamReader = null;
    appState.isStreaming = false;
  }
  init();
});

// ── Initialization ────────────────────────────────────────

async function init() {
  chatArea.innerHTML = '';
  hideAllInputs();

  appState.flowState = FLOW.INIT;
  appState.sessionId = null;
  appState.scenario = null;
  appState.category = null;
  appState.scriptType = null;
  appState.product = null;

  await advanceFlow();
}

document.addEventListener('DOMContentLoaded', init);
