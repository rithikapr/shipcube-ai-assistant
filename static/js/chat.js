const chatWindow = document.getElementById("chatWindow");
const chatInput = document.getElementById("chatBox");
const sendBtn = document.getElementById("sendBtn");
const faqList = document.getElementById("faqList");
const topSearch = document.getElementById("topSearch");
const topAsk = document.getElementById("topAsk");

function el(tag, cls, text) {
  const d = document.createElement(tag);
  if (cls) d.className = cls;
  if (text !== undefined) d.textContent = text;
  return d;
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
  }

  bubble.textContent = text || "";

  if (meta && meta.source) {
    const metaLine = el("div", "bubble-meta", "Source: " + meta.source);
    bubble.appendChild(metaLine);
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
sendBtn.addEventListener("click", sendMessage);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// optional top-search (currently you don't have these in HTML, so this is no-op)
if (topAsk) {
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

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[
      c
    ];
  });
}

// Render FAQ cards
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