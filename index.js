// index.js - Backend (admin panel eklendi, /admin route çalışıyor)
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const Groq = require('groq-sdk');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const multer = require('multer');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

if (!process.env.GROQ_API_KEY || !process.env.GEMINI_API_KEY) {
  console.error('❌ GROQ_API_KEY ve GEMINI_API_KEY eksik!');
  process.exit(1);
}

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static('public'));

const storage = multer.memoryStorage();
const upload = multer({ storage, limits: { fileSize: 10 * 1024 * 1024 } });

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

// --------------------- Veri yapıları ---------------------
const userData = new Map(); // userId -> { conversations, settings, name }
const allMessages = []; // { userId, userName, role, content, timestamp, convId }

function getUserData(userId) {
  if (!userData.has(userId)) {
    userData.set(userId, {
      conversations: new Map(),
      nextId: 1,
      name: null,
      settings: { personalityPrompt: 'Sen Vizzy AI, arkadaş canlısı bir asistansın. Her cevabında mutlaka uygun bir emoji kullan. 😊 Selam kanka diyene "Selam kanka! 😎" de. Türkçe konuş, samimi ol.' }
    });
  }
  return userData.get(userId);
}

function createConversation(userId, title = 'Yeni Sohbet') {
  const user = getUserData(userId);
  const id = String(user.nextId++);
  const now = new Date().toISOString();
  const conv = { id, title, messages: [], summary: '', createdAt: now, updatedAt: now };
  user.conversations.set(id, conv);
  return conv;
}

function getConversation(userId, convId) {
  return getUserData(userId).conversations.get(convId);
}

function generateChatTitle(message) {
  let clean = message.toLowerCase().replace(/[.,!?;:"'()]/g, '');
  const stopWords = new Set([
    've', 'veya', 'ile', 'ki', 'ben', 'sen', 'o', 'biz', 'siz', 'onlar',
    'bu', 'şu', 'bir', 'iki', 'üç', 'çok', 'az', 'daha', 'en',
    'için', 'kadar', 'gibi', 'mi', 'mu', 'mı', 'mü',
    'nasıl', 'neden', 'niye', 'nerede', 'hangi', 'ne', 'nereden',
    'benim', 'senin', 'onun', 'bizim', 'sizin', 'onların',
    'bana', 'sana', 'ona', 'bize', 'size', 'onlara',
    'a', 'an', 'the', 'and', 'or', 'but', 'so', 'for', 'of', 'to', 'in',
    'is', 'are', 'was', 'were', 'be', 'been', 'am', 'do', 'does', 'did',
    'have', 'has', 'had', 'can', 'could', 'will', 'would', 'should', 'may', 'might'
  ]);
  let words = clean.split(/\s+/);
  let filtered = words.filter(w => w.length > 2 && !stopWords.has(w));
  if (filtered.length === 0) filtered = words.slice(0, 3);
  let titleWords = filtered.slice(0, 3);
  let title = titleWords.map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  if (title.length > 40) title = title.slice(0, 40);
  if (!title || title.length < 2) title = 'Sohbet';
  return title;
}

async function updateConversation(userId, convId, userMsg, assistantMsg, imageUrl = null) {
  const conv = getConversation(userId, convId);
  if (!conv) return null;
  const content = imageUrl ? `[Resim: ${imageUrl}]\n${userMsg}` : userMsg;
  conv.messages.push({ role: 'user', content: userMsg });
  conv.messages.push({ role: 'assistant', content: assistantMsg });
  conv.updatedAt = new Date().toISOString();

  // Tüm mesajları global diziye ekle
  const userName = getUserData(userId).name || 'Anonim';
  allMessages.push({ userId, userName, role: 'user', content: userMsg, timestamp: new Date().toISOString(), convId });
  allMessages.push({ userId, userName, role: 'assistant', content: assistantMsg, timestamp: new Date().toISOString(), convId });

  if (conv.messages.length === 2 && conv.title === 'Yeni Sohbet') {
    const newTitle = generateChatTitle(userMsg);
    if (newTitle && newTitle !== 'Yeni Sohbet') conv.title = newTitle;
  }

  if (conv.messages.length > 30) {
    const old = conv.messages.slice(0, -20);
    const text = old.map(m => `${m.role}: ${m.content}`).join('\n').slice(0, 2000);
    conv.summary = `[Özet: ${text.slice(0, 500)}...]`;
    conv.messages = conv.messages.slice(-20);
  }
  conv.messages = truncateHistory(conv.messages, 20000);
  return conv;
}

function estimateTokens(text) { return Math.ceil((text || '').length / 4); }
function truncateHistory(messages, maxTokens) {
  let total = 0;
  const truncated = [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const t = estimateTokens(messages[i].content);
    if (total + t > maxTokens) break;
    total += t;
    truncated.unshift(messages[i]);
  }
  return truncated;
}

async function callAIWithFallback(userId, conv, userMessage, res) {
  const personality = getUserData(userId).settings.personalityPrompt;
  const recent = conv.messages.slice(-15);
  let full = [...recent];
  if (conv.summary) full.unshift({ role: 'system', content: conv.summary });
  full.unshift({ role: 'system', content: personality });
  full.push({ role: 'user', content: userMessage });
  full = truncateHistory(full, 28000);

  const msgs = full.map(m => ({
    role: m.role === 'assistant' ? 'assistant' : (m.role === 'user' ? 'user' : 'system'),
    content: m.content
  }));

  try {
    const stream = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: msgs,
      stream: true,
      temperature: 0.9,
      max_tokens: 4096,
    });
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    let fullResponse = '';
    for await (const chunk of stream) {
      const text = chunk.choices[0]?.delta?.content || '';
      fullResponse += text;
      res.write(text);
    }
    res.end();
    return fullResponse;
  } catch (err) {
    console.error('Groq error, fallback Gemini:', err.message);
    try {
      const prompt = msgs.map(m => `${m.role}: ${m.content}`).join('\n');
      const result = await geminiModel.generateContent({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.9, maxOutputTokens: 4096 },
      });
      const text = result.response.text();
      res.write(text);
      res.end();
      return text;
    } catch (e) {
      res.status(500).send('Hata oluştu, tekrar dene.');
      return '';
    }
  }
}

async function callVision(userId, conv, userMessage, imageBase64, res) {
  const personality = getUserData(userId).settings.personalityPrompt;
  const prompt = `${personality}\n\nKullanıcı: ${userMessage || 'Bu resimde ne var?'}\nResmi analiz et ve cevap ver. Emoji kullan.`;
  try {
    const result = await geminiModel.generateContent({
      contents: [{
        role: 'user',
        parts: [
          { text: prompt },
          { inlineData: { data: imageBase64, mimeType: 'image/jpeg' } }
        ]
      }],
      generationConfig: { temperature: 0.9, maxOutputTokens: 4096 },
    });
    const response = result.response.text();
    res.write(response);
    res.end();
    return response;
  } catch (err) {
    console.error('Vision error:', err);
    res.status(500).send('Resim analiz edilemedi.');
    return '';
  }
}

// Rate limiting ve queue (kısa tutuldu)
const rateStore = new Map();
function rateLimit(ip, userId) {
  const key = `${ip}_${userId}`;
  const now = Date.now();
  const rec = rateStore.get(key) || { count: 0, lastReset: now, cooldown: 0 };
  if (rec.cooldown > now) return { allowed: false, retryAfter: Math.ceil((rec.cooldown - now) / 1000) };
  if (now - rec.lastReset > 1000) { rec.count = 0; rec.lastReset = now; }
  rec.count++;
  if (rec.count > 10) { rec.cooldown = now + 2000; rateStore.set(key, rec); return { allowed: false, retryAfter: 2 }; }
  rateStore.set(key, rec);
  return { allowed: true, slowDown: rec.count > 6 };
}

const queue = [];
let processing = false;
async function processQueue() {
  if (processing || queue.length === 0) return;
  processing = true;
  while (queue.length) {
    const { resolve, reject, userId, conv, msg, res } = queue.shift();
    try {
      const r = await callAIWithFallback(userId, conv, msg, res);
      resolve(r);
    } catch (e) { reject(e); }
    await new Promise(r => setTimeout(r, 50));
  }
  processing = false;
}
function addToQueue(userId, conv, msg, res) {
  return new Promise((resolve, reject) => {
    queue.push({ resolve, reject, userId, conv, msg, res });
    processQueue();
  });
}

const cache = new Map();
function getCacheKey(msg, convId) { return `${convId}_${msg.slice(0, 100)}`; }
function getCached(key) {
  const e = cache.get(key);
  if (e && Date.now() - e.ts < 60000) return e.res;
  return null;
}
function setCache(key, res) {
  cache.set(key, { res, ts: Date.now() });
  if (cache.size > 100) {
    const oldest = [...cache.entries()].sort((a,b) => a[1].ts - b[1].ts)[0];
    cache.delete(oldest[0]);
  }
}

// --------------------- API Endpoints ---------------------
app.get('/api/conversations', (req, res) => {
  const { userId } = req.query;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  const convs = Array.from(getUserData(userId).conversations.values()).map(c => ({
    id: c.id, title: c.title, createdAt: c.createdAt, updatedAt: c.updatedAt
  }));
  res.json(convs);
});

app.post('/api/conversations', (req, res) => {
  const { userId, title } = req.body;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  const conv = createConversation(userId, title || 'Yeni Sohbet');
  res.json({ id: conv.id, title: conv.title });
});

app.get('/api/conversations/:id', (req, res) => {
  const { userId } = req.query;
  const { id } = req.params;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  const conv = getConversation(userId, id);
  if (!conv) return res.status(404).json({ error: 'Not found' });
  res.json({ messages: conv.messages, title: conv.title });
});

app.put('/api/conversations/:id', (req, res) => {
  const { userId, title } = req.body;
  const { id } = req.params;
  if (!userId || !title) return res.status(400).json({ error: 'userId and title required' });
  const conv = getConversation(userId, id);
  if (!conv) return res.status(404).json({ error: 'Not found' });
  conv.title = title;
  res.json({ success: true });
});

app.delete('/api/conversations/:id', (req, res) => {
  const { userId } = req.body;
  const { id } = req.params;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  const user = getUserData(userId);
  if (!user.conversations.has(id)) return res.status(404).json({ error: 'Not found' });
  user.conversations.delete(id);
  res.json({ success: true });
});

app.get('/api/settings', (req, res) => {
  const { userId } = req.query;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  res.json({ personalityPrompt: getUserData(userId).settings.personalityPrompt });
});

app.put('/api/settings', (req, res) => {
  const { userId, personalityPrompt } = req.body;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  if (personalityPrompt !== undefined) getUserData(userId).settings.personalityPrompt = personalityPrompt;
  res.json({ success: true });
});

app.post('/api/user/name', (req, res) => {
  const { userId, name } = req.body;
  if (!userId || !name) return res.status(400).json({ error: 'userId and name required' });
  const user = getUserData(userId);
  user.name = name;
  res.json({ success: true, name });
});

app.get('/api/user/name', (req, res) => {
  const { userId } = req.query;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  const user = getUserData(userId);
  res.json({ name: user.name || null });
});

app.post('/api/chat/:convId', async (req, res) => {
  const { message, userId } = req.body;
  const { convId } = req.params;
  const ip = req.ip || req.socket.remoteAddress;
  if (!message || !userId || !convId) return res.status(400).json({ error: 'Missing fields' });

  const rate = rateLimit(ip, userId);
  if (!rate.allowed) return res.status(429).json({ error: `Çok fazla istek. ${rate.retryAfter} saniye bekleyin.` });
  if (rate.slowDown) await new Promise(r => setTimeout(r, 800));

  let conv = getConversation(userId, convId);
  if (!conv) conv = createConversation(userId, 'Yeni Sohbet');

  const ckey = getCacheKey(message, convId);
  const cached = getCached(ckey);
  if (cached) {
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    await updateConversation(userId, convId, message, cached);
    return res.send(cached);
  }

  try {
    const response = await addToQueue(userId, conv, message, res);
    await updateConversation(userId, convId, message, response);
    setCache(ckey, response);
  } catch (err) {
    if (!res.headersSent) res.status(500).json({ error: 'AI servisi geçici olarak kullanılamıyor.' });
  }
});

app.post('/api/vision/:convId', upload.single('image'), async (req, res) => {
  const { message, userId } = req.body;
  const { convId } = req.params;
  const ip = req.ip || req.socket.remoteAddress;
  if (!userId || !convId) return res.status(400).json({ error: 'Missing fields' });
  if (!req.file) return res.status(400).json({ error: 'Resim yüklenmedi' });

  const rate = rateLimit(ip, userId);
  if (!rate.allowed) return res.status(429).json({ error: `Çok fazla istek. ${rate.retryAfter} saniye bekleyin.` });

  let conv = getConversation(userId, convId);
  if (!conv) conv = createConversation(userId, 'Yeni Sohbet');

  const imageBase64 = req.file.buffer.toString('base64');
  const userMessage = message || 'Bu resimde ne var?';

  try {
    const response = await callVision(userId, conv, userMessage, imageBase64, res);
    await updateConversation(userId, convId, userMessage, response, 'image');
  } catch (err) {
    if (!res.headersSent) res.status(500).json({ error: 'Görsel işlenirken hata oluştu.' });
  }
});

// Admin panel API'leri
app.post('/api/admin/login', (req, res) => {
  const { password } = req.body;
  if (password === '991') {
    res.json({ success: true });
  } else {
    res.status(401).json({ error: 'Şifre yanlış' });
  }
});

app.get('/api/admin/messages', (req, res) => {
  const auth = req.headers.authorization;
  if (auth !== 'Bearer admin_991') {
    return res.status(401).json({ error: 'Yetkisiz' });
  }
  const sorted = [...allMessages].sort((a,b) => new Date(a.timestamp) - new Date(b.timestamp));
  res.json(sorted);
});

// --------------------- ADMIN SAYFASI (GET /admin) ---------------------
app.get('/admin', (req, res) => {
  const html = `<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vizzy AI - Admin Paneli</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #1e1f22; color: #ececf1; padding: 20px; }
    .container { max-width: 1200px; margin: 0 auto; }
    h1 { margin-bottom: 20px; color: #10a37f; }
    .login-box { background: #2c2d31; padding: 30px; border-radius: 20px; max-width: 400px; margin: 100px auto; text-align: center; }
    .login-box input { width: 100%; padding: 12px; margin: 15px 0; border-radius: 40px; border: none; background: #1e1f22; color: white; font-size: 1rem; }
    .login-box button { background: #10a37f; color: white; border: none; padding: 12px 32px; border-radius: 40px; cursor: pointer; font-size: 1rem; }
    .messages-table { width: 100%; border-collapse: collapse; background: #2c2d31; border-radius: 16px; overflow: hidden; }
    .messages-table th, .messages-table td { padding: 12px; text-align: left; border-bottom: 1px solid #3c3d41; }
    .messages-table th { background: #10a37f; color: white; }
    .user-row { background: #1e1f22; }
    .assistant-row { background: #2c2d31; }
    .timestamp { font-size: 0.8rem; color: #aaa; }
    .error { color: #c72a2a; margin-top: 10px; }
  </style>
</head>
<body>
<div class="container" id="app">
  <div id="loginPanel" class="login-box">
    <h2>Admin Girişi</h2>
    <input type="password" id="passwordInput" placeholder="Şifre">
    <button id="loginBtn">Giriş Yap</button>
    <div id="loginError" class="error"></div>
  </div>
  <div id="adminPanel" style="display:none;">
    <h1>Vizzy AI - Tüm Mesajlar</h1>
    <div style="overflow-x: auto;">
      <table class="messages-table" id="messagesTable">
        <thead>
          <tr><th>Kullanıcı ID</th><th>Kullanıcı Adı</th><th>Rol</th><th>Mesaj</th><th>Zaman</th></tr>
        </thead>
        <tbody id="messagesBody"></tbody>
      </table>
    </div>
  </div>
</div>
<script>
  const loginPanel = document.getElementById('loginPanel');
  const adminPanel = document.getElementById('adminPanel');
  const passwordInput = document.getElementById('passwordInput');
  const loginBtn = document.getElementById('loginBtn');
  const loginError = document.getElementById('loginError');
  const messagesBody = document.getElementById('messagesBody');

  async function login() {
    const password = passwordInput.value.trim();
    if (!password) return;
    try {
      const res = await fetch('/api/admin/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password })
      });
      const data = await res.json();
      if (res.ok && data.success) {
        localStorage.setItem('admin_token', 'admin_991');
        loadMessages();
        loginPanel.style.display = 'none';
        adminPanel.style.display = 'block';
      } else {
        loginError.innerText = 'Şifre yanlış!';
      }
    } catch (err) {
      loginError.innerText = 'Bağlantı hatası';
    }
  }

  async function loadMessages() {
    try {
      const res = await fetch('/api/admin/messages', {
        headers: { 'Authorization': 'Bearer admin_991' }
      });
      if (!res.ok) throw new Error('Yetkisiz');
      const messages = await res.json();
      messagesBody.innerHTML = '';
      for (const msg of messages) {
        const row = document.createElement('tr');
        row.className = msg.role === 'user' ? 'user-row' : 'assistant-row';
        row.innerHTML = \`
          <td>\${escapeHtml(msg.userId)}</td>
          <td>\${escapeHtml(msg.userName)}</td>
          <td>\${msg.role === 'user' ? '👤 Kullanıcı' : '🤖 Bot'}</td>
          <td>\${escapeHtml(msg.content)}</td>
          <td class="timestamp">\${new Date(msg.timestamp).toLocaleString('tr-TR')}</td>
        \`;
        messagesBody.appendChild(row);
      }
    } catch (err) {
      console.error(err);
      messagesBody.innerHTML = '<tr><td colspan="5">Mesajlar yüklenemedi</td></tr>';
    }
  }

  function escapeHtml(str) {
    return str.replace(/[&<>]/g, m => m === '&' ? '&amp;' : m === '<' ? '&lt;' : '&gt;');
  }

  loginBtn.onclick = login;
  passwordInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') login(); });

  // Oturum kontrolü
  if (localStorage.getItem('admin_token') === 'admin_991') {
    loginPanel.style.display = 'none';
    adminPanel.style.display = 'block';
    loadMessages();
  }
</script>
</body>
</html>`;
  res.send(html);
});

app.get('/health', (req, res) => res.send('OK'));

function startServer(port) {
  const server = app.listen(port, () => console.log(`✅ Vizzy AI running on port ${port}`));
  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') startServer(port + 1);
    else console.error(err);
  });
}
startServer(PORT);
