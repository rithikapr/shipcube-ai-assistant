const chatWindow = document.getElementById("chatWindow");
const chatInput = document.getElementById("chatBox");
const sendBtn = document.getElementById("sendBtn");
const faqList = document.getElementById("faqList");
const topSearch = document.getElementById("topSearch");
const topAsk = document.getElementById("topAsk");
const feedbackState = {}; // { [messageId]: 1 or -1 }

function el(tag, cls, text) {
  const d = document.createElement(tag);
  if (cls) d.className = cls;
  if (text !== undefined) d.textContent = text;
  return d;
}
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
function scrollChat() {
  if (chatWindow) chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addBubble(text, role = "ai", meta) {
  const row = el(
    "div",
    `msg-row ${role === "user" ? "msg-user" : "msg-ai"}`
  );
  const avatar = el(
    "div",
    "avatar " + (role === "user" ? "avatar-user" : "avatar-ai")
  );
  const bubble = el(
    "div",
    "bubble " + (role === "user" ? "bubble-user" : "bubble-ai")
  );

  if (role === "user") {
    avatar.textContent = "U";
  } else {
    avatar.style.backgroundImage = "url('/static/images/shipcube_logo.png')";
    avatar.style.backgroundSize = "cover";
  }

  const textDiv = el("div", "bubble-text");
  textDiv.innerHTML = escapeHtml(text || "");
  bubble.appendChild(textDiv);

  if (meta && meta.source) {
    const metaLine = el("div", "bubble-meta", "Source: " + meta.source);
    bubble.appendChild(metaLine);
  }
  if (role === "ai") {
    const msgId =
      "msg-" + Date.now() + "-" + Math.floor(Math.random() * 100000);
    row.dataset.msgId = msgId;

    // store question/answer later via dataset if you want; for now only answer.
    row.dataset.answer = text || "";

    const feedbackDiv = el("div", "feedback");
    const label = el("span", "feedback-label", "Was this helpful?");
    const upBtn = el("button", "thumb thumb-up", "👍");
    const downBtn = el("button", "thumb thumb-down", "👎");

    upBtn.setAttribute("data-value", "1");
    downBtn.setAttribute("data-value", "-1");

    feedbackDiv.appendChild(label);
    feedbackDiv.appendChild(upBtn);
    feedbackDiv.appendChild(downBtn);

    bubble.appendChild(feedbackDiv);
  }

  if (role === "user") {
    row.appendChild(bubble);
    row.appendChild(avatar);
  } else {
    row.appendChild(avatar);
    row.appendChild(bubble);
  }

  chatWindow.appendChild(row);
  scrollChat();
}

// typing indicator
let typingEl = null;
function showTyping(show = true) {
  if (show) {
    if (typingEl) return;
    typingEl = el("div", "msg-row msg-ai typing-row");
    const avatar = el("div", "avatar avatar-ai");
    avatar.style.backgroundImage = "url('/static/images/shipcube_logo.png')";
    const bubble = el("div", "bubble bubble-ai");
    bubble.textContent = "Thinking...";
    typingEl.appendChild(avatar);
    typingEl.appendChild(bubble);
    chatWindow.appendChild(typingEl);
    scrollChat();
  } else {
    if (typingEl) {
      typingEl.remove();
      typingEl = null;
    }
  }
}

async function askServer(payload) {
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      const txt = await res.text();
      return { ok: false, error: txt };
    }
    const json = await res.json();
    return { ok: true, data: json };
  } catch (err) {
    return { ok: false, error: String(err) };
  }
}

async function sendMessage() {
  const text = (chatInput.value || "").trim();
  if (!text) return;

  addBubble(text, "user");
  chatInput.value = "";
  showTyping(true);

  const resp = await askServer({ query: text });
  showTyping(false);

  if (!resp.ok) {
    addBubble("Sorry — couldn't reach the server. Try again.", "ai");
    console.error("ask error", resp.error);
    return;
  }
  const body = resp.data && resp.data.response;
  if (!body) {
    addBubble("No response from server.", "ai");
    return;
  }

  let answerText = null,
    source = null;
  if (typeof body === "string") {
    answerText = body;
    source = "generated";
  } else if (typeof body === "object") {
    answerText = body.answer || body.text || "";
    source = body.source || "retrieval_or_generated";
  }

  if (
    source === "auth_required" ||
    (answerText &&
      answerText.toLowerCase().includes("please log in to view order-specific"))
  ) {
    addBubble(
      answerText || "Please log in to view this info.",
      "ai",
      { source }
    );
    return;
  }

  addBubble(answerText || "No answer found.", "ai", { source });
}

// wire send
if (sendBtn) {
  sendBtn.addEventListener("click", sendMessage);
}
if (chatInput) {
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}
// optional top-search
if (topAsk && topSearch) {
  topAsk.addEventListener("click", () => {
    chatInput.value = topSearch.value;
    sendMessage();
  });
  topSearch.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      chatInput.value = topSearch.value;
      sendMessage();
    }
  });
}

// --------- FAQ helpers (right-hand pane) ---------
function renderFAQ(items) {
  if (!faqList) return;
  faqList.innerHTML = "";

  if (!items || !items.length) {
    faqList.innerHTML = '<div class="faq-empty">No FAQ available.</div>';
    return;
  }

  items.slice(0, 10).forEach((item) => {
    const card = el("div", "faq-item");
    const q = el("div", "faq-q", item.question || "");
    const ansText = item.answer || "";
    const snippet =
      ansText.length > 220 ? ansText.slice(0, 220).trim() + "…" : ansText;

    const a = el("div", "faq-a", snippet);
    card.appendChild(q);
    card.appendChild(a);

    // click a card to move its question into the chat box
    card.addEventListener("click", () => {
      chatInput.value = item.question || "";
      chatInput.focus();
    });

    faqList.appendChild(card);
  });
}

// Load FAQ for a given category/tag
async function loadCategoryFAQ(cat) {
  if (!faqList) return;
  faqList.innerHTML = '<div class="faq-empty">Loading…</div>';

  try {
    const res = await fetch(`/faq/${encodeURIComponent(cat)}`);
    const data = await res.json();
    if (!data.ok) throw new Error("not ok");
    renderFAQ(data.items || []);
  } catch (err) {
    console.error("FAQ cat error", err);
    faqList.innerHTML =
      '<div class="faq-empty">Failed to load FAQ.</div>';
  }
}

// category buttons: highlight + fetch category FAQ
document.querySelectorAll(".category").forEach((btn) => {
  btn.addEventListener("click", () => {
    const cat = btn.dataset.cat;

    document
      .querySelectorAll(".category")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    loadCategoryFAQ(cat);
  });
});

// init on load: select "about" and load its FAQ
window.addEventListener("load", () => {
  const defaultBtn = document.querySelector('.category[data-cat="about"]');
  if (defaultBtn) {
    defaultBtn.classList.add("active");
    loadCategoryFAQ("about");
  }
});

  // Global click handler for thumbs (event delegation)
  document.addEventListener('click', function (e) {
    const btn = e.target.closest('.thumb');
    if (!btn) return;

    const msgRow = btn.closest('.msg-row');
    if (!msgRow) return;

    const msgId = msgRow.dataset.msgId;
    const value = parseInt(btn.getAttribute("data-value"), 10); // 1 or -1
    if (!msgId || (value !== 1 && value !== -1)) return;

    const upBtn = msgRow.querySelector('.thumb-up');
    const downBtn = msgRow.querySelector('.thumb-down');

    if (!upBtn || !downBtn) return;

    // Radio-button style toggle
    if (value === 1) {
      upBtn.classList.add('active');
      downBtn.classList.remove('active');
    } else {
      downBtn.classList.add('active');
      upBtn.classList.remove('active');
    }

    feedbackState[msgId] = value;

    // Send to backend
    const payload = {
      message_id: msgId,
      rating: value,
      question: msgRow.dataset.question || null,
      answer: msgRow.dataset.answer || null,
      model: "gemini-2.5-flash-lite" // or read from a global var if you like
    };

    sendFeedback(payload);
  });

  function sendFeedback(payload) {
    fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(err => {
      console.error('Feedback send error:', err);
    });
  }