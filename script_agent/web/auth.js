/* =========================================================
   ScriptAgent - Auth Page
   ========================================================= */

const CLIENT_CONFIG_KEY = 'script_agent_frontend_config_v1';
const AUTH_TOKEN_KEY = 'script_agent_auth_token';

const toastContainer = document.getElementById('toastContainer');
const tabLogin = document.getElementById('tabLogin');
const tabRegister = document.getElementById('tabRegister');
const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');
const authEnvTag = document.getElementById('authEnvTag');

function showToast(message, type, durationMs) {
  type = type || 'info';
  durationMs = durationMs || 2600;
  var icons = { success: '✅', error: '❌', info: 'ℹ️' };
  var toast = document.createElement('div');
  toast.className = 'toast toast-' + type;
  toast.innerHTML =
    '<span class="toast-icon">' + (icons[type] || icons.info) + '</span>' +
    '<span>' + message + '</span>';
  toastContainer.appendChild(toast);
  setTimeout(function() {
    toast.classList.add('toast-exit');
    setTimeout(function() { toast.remove(); }, 260);
  }, durationMs);
}

function showLoginTab() {
  tabLogin.classList.add('auth-tab-active');
  tabRegister.classList.remove('auth-tab-active');
  loginForm.classList.remove('hidden');
  registerForm.classList.add('hidden');
}

function showRegisterTab() {
  tabRegister.classList.add('auth-tab-active');
  tabLogin.classList.remove('auth-tab-active');
  registerForm.classList.remove('hidden');
  loginForm.classList.add('hidden');
}

function saveAuthToken(token) {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
  var cfg = {};
  try {
    cfg = JSON.parse(localStorage.getItem(CLIENT_CONFIG_KEY) || '{}');
  } catch (_) {
    cfg = {};
  }
  cfg.bearerToken = token;
  localStorage.setItem(CLIENT_CONFIG_KEY, JSON.stringify(cfg));
}

async function loadFrontendConfig() {
  try {
    var res = await fetch('/api/v1/frontend-config');
    if (!res.ok) throw new Error('fetch failed');
    var cfg = await res.json();
    var lines = [];
    lines.push('环境: ' + (cfg.env || 'unknown'));
    lines.push(cfg.auth_required ? '需要登录鉴权' : '开发模式可免登');
    authEnvTag.textContent = lines.join(' | ');
    if (cfg.registration_enabled === false) {
      tabRegister.classList.add('hidden');
    }
  } catch (_) {
    authEnvTag.textContent = '环境: unknown';
  }
}

async function doLogin(username, password) {
  var res = await fetch('/api/v1/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: username, password: password }),
  });
  if (!res.ok) {
    var text = await res.text();
    throw new Error(text || '登录失败');
  }
  return await res.json();
}

async function doRegister(username, password, tenantId) {
  var res = await fetch('/api/v1/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      username: username,
      password: password,
      tenant_id: tenantId,
      role: 'user',
    }),
  });
  if (!res.ok) {
    var text = await res.text();
    throw new Error(text || '注册失败');
  }
  return await res.json();
}

loginForm.addEventListener('submit', async function(e) {
  e.preventDefault();
  var username = document.getElementById('loginUsername').value.trim();
  var password = document.getElementById('loginPassword').value;
  try {
    var result = await doLogin(username, password);
    saveAuthToken(result.access_token || '');
    showToast('登录成功，正在跳转', 'success');
    setTimeout(function() { window.location.href = '/'; }, 500);
  } catch (err) {
    showToast('登录失败: ' + err.message, 'error');
  }
});

registerForm.addEventListener('submit', async function(e) {
  e.preventDefault();
  var username = document.getElementById('registerUsername').value.trim();
  var password = document.getElementById('registerPassword').value;
  var tenantId = document.getElementById('registerTenantId').value.trim() || 'tenant_dev';
  try {
    await doRegister(username, password, tenantId);
    showToast('注册成功，请登录', 'success');
    showLoginTab();
  } catch (err) {
    showToast('注册失败: ' + err.message, 'error');
  }
});

tabLogin.addEventListener('click', showLoginTab);
tabRegister.addEventListener('click', showRegisterTab);

document.addEventListener('DOMContentLoaded', function() {
  loadFrontendConfig();
});
