// The app: a shelf of books on the left, tabs on the right (Home, Search, one per open book).
import { ICON } from "/static/icons.js";
import * as pdfjs from "/static/vendor/pdf.min.mjs";

pdfjs.GlobalWorkerOptions.workerSrc = "/static/vendor/pdf.worker.min.mjs";

const out = document.getElementById("out");
const box = document.getElementById("q");
const state = document.getElementById("state");
const list = document.getElementById("book-list");
const filter = document.getElementById("filter");
const sideCount = document.getElementById("side-count");
const tabs = document.getElementById("tabs");
const views = document.getElementById("views");
let timer = null;
let generation = 0;
let books = [];
let marks = new Set();           // bookmarked book ids, as strings
let positions = {};              // book id -> last page read
const open = new Map();          // book id -> {title, doc, leaves, page, zoom, scale, io, save}
let active = "home";

for (const el of document.querySelectorAll("[data-icon]")) el.innerHTML = ICON[el.dataset.icon];
document.getElementById("nav-toggle").addEventListener("click", () => document.body.classList.toggle("nav-closed"));

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}

/* ---- the shelf: fetched once, filtered in the browser ------------------------ */

async function loadBooks() {
  books = await (await fetch("/api/books")).json();
  drawBooks("");
  drawHome();
}

function drawBooks(needle) {
  const query = needle.trim().toLowerCase();
  const shown = query
    ? books.filter((b) => (b.title + " " + (b.subtitle || "") + " " + (b.authors || []).join(" ")).toLowerCase().includes(query))
    : books;
  sideCount.textContent = query ? shown.length + " / " + books.length : books.length;
  // A row holds two buttons (open, bookmark), so the row itself is a plain box.
  list.innerHTML = shown.map((b) => {
    const on = marks.has(String(b.id));
    return '<div class="bk">'
      + '<button class="bkopen" data-id="' + b.id + '">'
      + '<span class="t">' + escapeHtml(b.title) + '</span>'
      + '<span class="s">' + sub(b) + '</span></button>'
      + '<button class="star" data-star="' + b.id + '" aria-pressed="' + on + '" title="Bookmark">'
      + (on ? ICON.starOn : ICON.star) + '</button></div>';
  }).join("");
}

// byline, year, pages, skipping what a book does not have.
function sub(b) {
  const bits = [];
  const by = byline(b.authors);
  if (by) bits.push(escapeHtml(by));
  if (b.year) bits.push(b.year);
  if (b.pages) bits.push(b.pages + " pp");
  return bits.join(", ");
}

function byline(authors) {
  if (!authors || !authors.length) return "";
  if (authors.length === 1) return authors[0];
  if (authors.length === 2) return authors[0] + " and " + authors[1];
  return authors[0] + " et al.";
}

filter.addEventListener("input", () => drawBooks(filter.value));
list.addEventListener("click", (event) => {
  const star = event.target.closest("[data-star]");
  if (star) return toggleMark(star.dataset.star);
  const item = event.target.closest(".bkopen");
  if (item) openBook(item.dataset.id, null);
});

/* ---- bookmarks: one id per write, merged on the server ----------------------- */

async function loadMarks() {
  const ids = await (await fetch("/api/bookmarks")).json().catch(() => []);
  marks = new Set(ids.map(String));
}

async function toggleMark(id) {
  id = String(id);
  const on = !marks.has(id);
  if (on) marks.add(id); else marks.delete(id);
  drawBooks(filter.value);
  drawHome();
  const ids = await (await fetch("/api/bookmark?id=" + encodeURIComponent(id), {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ on }),
  }).catch(() => null))?.json().catch(() => null) || null;
  if (ids) { marks = new Set(ids.map(String)); drawBooks(filter.value); drawHome(); }
}

// Home: the bookmarked books, each opening at the page last read.
function drawHome() {
  const host = document.getElementById("home");
  const shelf = books.filter((b) => marks.has(String(b.id)));
  if (!shelf.length) {
    host.innerHTML = '<div class="empty">No bookmarks</div>';
    return;
  }
  host.innerHTML = shelf.map((b) => {
    const at = positions[String(b.id)];
    return '<div class="card"><div class="body">'
      + '<div class="t">' + escapeHtml(b.title) + '</div>'
      + '<div class="s">' + sub(b) + '</div>'
      + '<div class="go"><button class="act" data-id="' + b.id + '">' + ICON.book
      + (at ? "p. " + at : "Open") + '</button></div>'
      + '</div><button class="star" data-star="' + b.id + '" aria-pressed="true" title="Bookmark">'
      + ICON.starOn + '</button></div>';
  }).join("");
}

document.getElementById("home").addEventListener("click", (event) => {
  const star = event.target.closest("[data-star]");
  if (star) return toggleMark(star.dataset.star);
  const go = event.target.closest("[data-id]");
  if (go) openBook(go.dataset.id, null);
});

/* ---- tabs and the reader ------------------------------------------------------ */

positions = await (await fetch("/api/positions")).json().catch(() => ({}));
await loadMarks();

// The open tabs (book, page, zoom) and the active one, kept in this browser so a reload restores them.
const TABS = "dirag.tabs";
const savedTabs = (() => { try { return JSON.parse(localStorage.getItem(TABS)) || {}; } catch (e) { return {}; } })();

function saveTabs() {
  const list = [...open.entries()].map(([id, st]) => ({ id, title: st.title, page: st.page, zoom: st.zoom }));
  try { localStorage.setItem(TABS, JSON.stringify({ active, tabs: list })); } catch (e) {}
}

// Reopen the saved tabs in order, each while it is visible so it lays out at the pane's width.
// A tab is reopened only if its id still names the same book (ids change when an index is rebuilt).
// A click or key press while this runs keeps the tab the user chose.
async function restoreTabs() {
  let acted = false;
  const act = () => { acted = true; };
  document.addEventListener("pointerdown", act, { capture: true, once: true });
  document.addEventListener("keydown", act, { capture: true, once: true });
  for (const t of savedTabs.tabs || []) {
    const meta = books.find((b) => String(b.id) === String(t.id));
    if (!meta || meta.title !== t.title || open.has(String(t.id))) continue;
    const keep = acted ? active : null;
    await openBook(t.id, t.page, null, t.zoom).catch(() => {});
    if (keep && active === String(t.id)) activate(keep);
  }
  document.removeEventListener("pointerdown", act, { capture: true });
  document.removeEventListener("keydown", act, { capture: true });
  const key = savedTabs.active;
  if (!acted && key && (open.has(key) || document.getElementById("view-" + key))) activate(key);
}

function activate(key) {
  active = key;
  saveTabs();
  for (const v of views.children) v.hidden = v.id !== "view-" + key;
  for (const t of tabs.children) t.setAttribute("aria-selected", String(t.dataset.key === key));
  // A hidden view can lose its scroll offset; put a book back where it was.
  const st = open.get(key);
  const host = document.getElementById("pages-" + key);
  if (st?.top != null && host && host.scrollTop !== st.top) host.scrollTop = st.top;
  if (key === "search") box.focus();
}

function drawTabs() {
  tabs.innerHTML = '<button class="tab" data-key="home">' + ICON.home + '<span class="lbl">Home</span></button>'
    + '<button class="tab" data-key="search">' + ICON.search + '<span class="lbl">Search</span></button>'
    + [...open.entries()].map(([id, st]) =>
      '<button class="tab" data-key="' + id + '"><span class="lbl">' + escapeHtml(st.title)
      + '</span><span class="x" data-close="' + id + '">' + ICON.close + '</span></button>').join("");
  activate(active);
}

tabs.addEventListener("click", (e) => {
  const x = e.target.closest("[data-close]");
  if (x) { e.stopPropagation(); return closeBook(x.dataset.close); }
  const t = e.target.closest(".tab");
  if (t) activate(t.dataset.key);
});

function closeBook(id) {
  const st = open.get(id);
  if (st) { st.doc?.destroy(); open.delete(id); }
  document.getElementById("view-" + id)?.remove();
  if (active === id) active = "search";
  drawTabs();
}

// Open a book as a tab at `page`, else at the page last read. An open book only jumps.
// `rects` (PDF points) mark a passage or quote on that page and the view scrolls to it.
async function openBook(id, page, rects, zoom) {
  id = String(id);
  if (open.has(id)) {
    activate(id);
    if (page) { setMark(id, Number(page), rects); goTo(id, Number(page), false, rects); }
    return;
  }
  const meta = books.find((b) => String(b.id) === id) || { title: "", pages: 0 };
  const start = Number(page) || positions[id] || 1;
  const view = document.createElement("div");
  view.className = "view";
  view.id = "view-" + id;
  view.innerHTML = '<div class="rbar"><span class="who">' + escapeHtml(meta.title) + '</span>'
    + '<button class="ib" data-zoom="' + id + '|-1" title="Zoom out">' + ICON.zoomOut + '</button>'
    + '<span class="of" id="zoom-' + id + '">100%</span>'
    + '<button class="ib" data-zoom="' + id + '|1" title="Zoom in">' + ICON.zoomIn + '</button>'
    + '<span class="sep"></span>'
    + '<button class="ib" data-step="' + id + '|-1" title="Previous page">' + ICON.prev + '</button>'
    + '<button class="ib" data-step="' + id + '|1" title="Next page">' + ICON.next + '</button>'
    + '<input type="text" inputmode="numeric" id="at-' + id + '" value="' + start + '" aria-label="Page">'
    + '<span class="of" id="of-' + id + '"></span>'
    + '</div><div class="pages" id="pages-' + id + '"><div class="rload">...</div></div>';
  views.appendChild(view);
  open.set(id, { title: meta.title, page: start, zoom: zoom || 1, mark: page && rects?.length ? { page: Number(page), rects } : null });
  drawTabs();
  activate(id);

  const doc = await pdfjs.getDocument({ url: "/book?id=" + id }).promise;
  const st = open.get(id);
  if (!st) { doc.destroy(); return; }
  st.doc = doc;
  document.getElementById("of-" + id).textContent = "/ " + doc.numPages;
  document.getElementById("zoom-" + id).textContent = Math.round(st.zoom * 100) + "%";
  await layout(id);
  goTo(id, start, true, page ? rects : null);
}

// One marked passage per open book, drawn as boxes over its page.
function setMark(id, page, rects) {
  const st = open.get(id);
  if (!st) return;
  st.mark = rects?.length ? { page, rects } : null;
  for (const box of document.querySelectorAll("#pages-" + id + " .box")) box.remove();
  if (st.mark && st.leaves) paintMark(id, st.leaves[page - 1]);
}

function paintMark(id, leaf) {
  const st = open.get(id);
  if (!st?.mark || !leaf || Number(leaf.dataset.page) !== st.mark.page || leaf.querySelector(".box")) return;
  const w = Number(leaf.dataset.w) || st.base.width;
  const h = Number(leaf.dataset.h) || st.base.height;
  for (const [x0, y0, x1, y1] of st.mark.rects) leaf.appendChild(boxAt(x0, y0, x1, y1, w, h));
}

function boxAt(x0, y0, x1, y1, w, h) {
  const box = document.createElement("div");
  box.className = "box";
  box.style.left = (100 * x0 / w) + "%";
  box.style.top = (100 * y0 / h) + "%";
  box.style.width = (100 * (x1 - x0) / w) + "%";
  box.style.height = (100 * (y1 - y0) / h) + "%";
  return box;
}

// One placeholder leaf per page, sized from page 1; each leaf takes its own size when it renders.
// Zoom 1 fits the pane width.
async function layout(id, keepPage) {
  const st = open.get(id);
  const host = document.getElementById("pages-" + id);
  const base = (await st.doc.getPage(1)).getViewport({ scale: 1 });
  st.base = { width: base.width, height: base.height };
  const width = Math.min(host.clientWidth - 28, 900) * (st.zoom || 1);
  st.scale = width / base.width;
  if (st.io) st.io.disconnect();
  host.innerHTML = "";
  for (let p = 1; p <= st.doc.numPages; p++) {
    const leaf = document.createElement("div");
    leaf.className = "leaf";
    leaf.dataset.page = p;
    leaf.style.width = width + "px";
    leaf.style.height = Math.round(width * base.height / base.width) + "px";
    leaf.innerHTML = '<span class="num">' + p + '</span>';
    host.appendChild(leaf);
  }
  st.leaves = [...host.children];
  if (st.mark) paintMark(id, st.leaves[st.mark.page - 1]);
  // Render pages near the viewport; free the canvases of pages that leave it.
  st.io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) render(id, Number(e.target.dataset.page));
      else discard(e.target);
    }
  }, { root: host, rootMargin: "200% 0px" });
  st.leaves.forEach((l) => st.io.observe(l));
  if (!st.bound) { host.addEventListener("scroll", () => onScroll(id), { passive: true }); st.bound = true; }
  if (keepPage) goTo(id, keepPage, true);
}

const ZOOMS = [0.75, 1, 1.25, 1.5, 2, 2.5];
async function zoom(id, dir) {
  const st = open.get(id);
  if (!st || !st.doc) return;
  const at = ZOOMS.indexOf(st.zoom || 1);
  const next = ZOOMS[Math.max(0, Math.min(ZOOMS.length - 1, (at < 0 ? 1 : at) + dir))];
  if (next === st.zoom) return;
  st.zoom = next;
  saveTabs();
  document.getElementById("zoom-" + id).textContent = Math.round(next * 100) + "%";
  await layout(id, st.page || 1);
}

async function render(id, p) {
  const st = open.get(id);
  if (!st || !st.doc) return;
  const leaf = st.leaves[p - 1];
  if (!leaf || leaf.querySelector("canvas")) return;
  const page = await st.doc.getPage(p);
  const view = page.getViewport({ scale: st.scale });
  const unit = page.getViewport({ scale: 1 });
  leaf.dataset.w = unit.width;
  leaf.dataset.h = unit.height;
  leaf.style.height = Math.round(view.height) + "px";
  const canvas = document.createElement("canvas");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(view.width * dpr);
  canvas.height = Math.round(view.height * dpr);
  leaf.prepend(canvas);
  await page.render({
    canvasContext: canvas.getContext("2d", { alpha: false }),
    viewport: page.getViewport({ scale: st.scale * dpr }),
  }).promise;
  // The page's text, invisible and placed over the image, so it can be selected and copied.
  if (!canvas.isConnected) return;
  const text = document.createElement("div");
  text.className = "textLayer";
  leaf.style.setProperty("--scale-factor", st.scale);
  canvas.after(text);
  await new pdfjs.TextLayer({ textContentSource: page.streamTextContent(), container: text, viewport: view }).render().catch(() => {});
  const end = document.createElement("div");
  end.className = "endOfContent";
  text.append(end);
  text.addEventListener("pointerdown", () => text.classList.add("selecting"));
}

document.addEventListener("pointerup", () => {
  for (const el of document.querySelectorAll(".textLayer.selecting")) el.classList.remove("selecting");
});

function discard(leaf) {
  const canvas = leaf.querySelector("canvas");
  if (canvas) { canvas.width = canvas.height = 0; canvas.remove(); }
  leaf.querySelector(".textLayer")?.remove();
}

// Scroll to page `p`, or to the first of `rects` on it with some room above.
function goTo(id, p, instant, rects) {
  const st = open.get(id);
  if (!st || !st.leaves) return;
  const leaf = st.leaves[Math.max(1, Math.min(st.doc.numPages, p)) - 1];
  if (!leaf) return;
  const host = document.getElementById("pages-" + id);
  const y = rects?.length ? Math.max(0, Math.min(...rects.map((r) => r[1])) * st.scale - 80) : 0;
  host.scrollTo({ top: leaf.offsetTop - host.offsetTop + y, behavior: instant ? "auto" : "smooth" });
  if (instant) onScroll(id);
}

// The current page is the topmost one on screen, saved 700 ms after scrolling stops.
// A hidden view has no layout, so its scroll events are ignored.
function onScroll(id) {
  const st = open.get(id);
  const host = document.getElementById("pages-" + id);
  if (!st || !st.leaves || !host || !host.clientHeight) return;
  st.top = host.scrollTop;
  const top = host.offsetTop + host.scrollTop + 40;
  let page = 1;
  for (const leaf of st.leaves) { if (leaf.offsetTop <= top) page = Number(leaf.dataset.page); else break; }
  if (page === st.page) return;
  st.page = page;
  saveTabs();
  const at = document.getElementById("at-" + id);
  if (at) at.value = page;
  clearTimeout(st.save);
  st.save = setTimeout(() => {
    positions[id] = page;
    fetch("/api/position?id=" + encodeURIComponent(id), {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ page }),
    }).catch(() => {});
  }, 700);
}

views.addEventListener("click", (e) => {
  const zin = e.target.closest("[data-zoom]");
  if (zin) { const [id, dir] = zin.dataset.zoom.split("|"); return zoom(id, Number(dir)); }
  const step = e.target.closest("[data-step]");
  if (!step) return;
  const [id, dir] = step.dataset.step.split("|");
  goTo(id, (open.get(id)?.page || 1) + Number(dir));
});
views.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || !e.target.id.startsWith("at-")) return;
  goTo(e.target.id.slice(3), Number(e.target.value) || 1);
});

/* ---- library: the folder and the indexing job ---------------------------------- */

const $ = (id) => document.getElementById(id);
let lib = {};
let pickAt = null;
let polling = null;

async function loadLibrary(scan) {
  lib = await (await fetch("/api/library" + (scan ? "?scan=1" : ""))).json().catch(() => ({}));
  $("lib-path").textContent = lib.path || "No folder";
  const bits = [];
  if (lib.books != null) bits.push(lib.books + " books");
  if (lib.no_text) bits.push(lib.no_text + " without text");
  $("lib-counts").innerHTML = bits.join(", ");
  const found = [];
  if (lib.new) found.push(lib.new + " new");
  if (lib.changed) found.push(lib.changed + " changed");
  if (lib.gone) found.push(lib.gone + " gone");
  $("job-scan").innerHTML = lib.new == null ? "" : "Scan: " + (found.join(", ") || "up to date");
  $("lib-change-row").hidden = !!lib.fixed;
  const job = await (await fetch("/api/job")).json().catch(() => ({}));
  drawJob(job);
}

const minutes = (s) => s < 60 ? "<1 min" : Math.round(s / 60) + " min";

function drawJob(job) {
  const run = job.state === "running";
  const step = job.step || {};
  const inBook = step.name === "embed" && step.of ? (step.size || 0) * step.n / step.of : 0;
  const frac = job.bytes_total ? (job.bytes_done + inBook) / job.bytes_total : 0;
  $("job-run").hidden = $("job-stop").hidden = !run;
  $("lib-scan").hidden = $("job-reindex").hidden = run;
  $("job-update").hidden = run || !(lib.new || lib.changed || lib.gone);
  $("lib-scan").disabled = $("job-update").disabled = $("job-reindex").disabled = !lib.path;
  $("lib-change").disabled = run;
  $("job-bar").value = frac;
  $("side-progress").hidden = !run;
  $("side-progress").firstElementChild.style.width = (100 * frac) + "%";
  if (run) {
    const bits = [job.done + " / " + (job.total || "-")];
    if (job.eta != null) bits.push(minutes(job.eta) + " left");
    $("job-line").innerHTML = bits.map(escapeHtml).join(", ");
    const book = [];
    if (job.current) book.push(job.current);
    if (step.name) book.push((step.name === "read" ? "pages " : "embedded ") + step.n + " / " + step.of);
    if (step.started) book.push(minutes(Date.now() / 1000 - step.started));
    $("job-book").innerHTML = book.map(escapeHtml).join(", ");
  }
  $("job-last").textContent = run ? "" : lastRun(job);
  $("job-scan").hidden = run;
  if (run && !polling) polling = setInterval(poll, 1500);
  if (!run && polling) { clearInterval(polling); polling = null; refresh(); }
}

function lastRun(job) {
  if (!job.state) return "";
  if (job.state === "failed") return "Failed: " + (job.error || "");
  const c = job.counts || {};
  const bits = [(job.state === "stopped" ? "Stopped" : "Done") + ": " + ((c.ok || 0) + (c.rechunk || 0)) + " indexed"];
  if (c.removed) bits.push(c.removed + " removed");
  if (c.error) bits.push(c.error + " failed");
  return bits.join(", ");
}

async function poll() {
  const job = await (await fetch("/api/job")).json().catch(() => null);
  if (job) drawJob(job);
}

async function refresh() {
  positions = await (await fetch("/api/positions")).json().catch(() => ({}));
  await loadMarks();
  await loadBooks();
  await loadLibrary();
}

async function post(url, body) {
  return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
}

async function startJob(action) {
  const res = await post("/api/job", { action });
  if (res.ok) drawJob(await res.json());
}

$("lib-open").addEventListener("click", () => { activate("library"); loadLibrary(); });
$("lib-scan").addEventListener("click", async () => {
  $("lib-scan").disabled = true;
  $("job-scan").textContent = "Scanning";
  await loadLibrary(true);
});
$("job-update").addEventListener("click", () => startJob("update"));
$("job-reindex").addEventListener("click", () => { if (confirm("Reindex every book?")) startJob("reindex"); });
$("job-stop").addEventListener("click", async () => drawJob(await (await post("/api/job/stop")).json()));

// The folder picker lists subfolders on the server, under its browse root.
async function browse(path) {
  const data = await (await fetch("/api/dirs?path=" + encodeURIComponent(path || ""))).json().catch(() => null);
  if (!data) return;
  pickAt = data;
  $("pick-path").textContent = data.path;
  $("pick-up").disabled = !data.parent;
  $("pick-pdfs").textContent = data.pdfs ? data.pdfs + " PDFs" : "";
  $("pick-dirs").innerHTML = data.dirs.map((d) =>
    '<button class="dir" data-dir="' + escapeHtml(d) + '">' + ICON.folder + escapeHtml(d) + '</button>').join("");
}

$("lib-change").addEventListener("click", () => {
  $("picker").hidden = !$("picker").hidden;
  if (!$("picker").hidden) browse(lib.path);
});
$("pick-up").addEventListener("click", () => pickAt?.parent && browse(pickAt.parent));
$("pick-dirs").addEventListener("click", (e) => {
  const dir = e.target.closest("[data-dir]");
  if (dir) browse(pickAt.path.replace(/\/$/, "") + "/" + dir.dataset.dir);
});
$("pick-use").addEventListener("click", async () => {
  if (!pickAt) return;
  const res = await post("/api/library", { path: pickAt.path });
  if (!res.ok) return;
  $("picker").hidden = true;
  for (const id of [...open.keys()]) closeBook(id);
  activate("library");
  await refresh();
});

/* ---- search: matches while typing, Enter to search, Ctrl+Enter for an AI answer ----- */

const mode = $("mode");
const rerank = $("rerank");

// The two choices are remembered per browser; storage can be unavailable, which only loses that.
try {
  mode.value = localStorage.getItem("dirag.mode") || "hybrid";
  rerank.value = localStorage.getItem("dirag.rerank") || "off";
} catch (e) {}
for (const select of [mode, rerank]) {
  select.addEventListener("change", () => {
    try { localStorage.setItem("dirag." + select.id, select.value); } catch (e) {}
    if (lastAsk !== null && box.value.trim().length >= 2) run(lastAsk);
  });
}

// The LLM rerank and the AI answer are offered only when the server has a language model configured.
fetch("/api/features").then((r) => r.json()).then((f) => {
  rerank.querySelector('[value="llm"]').disabled = !f.llm;
  $("ask").hidden = !f.llm;
  if (!f.llm && rerank.value === "llm") rerank.value = "off";
}).catch(() => {});

/* Typing fills the matches box under the field: quick exact-word matches, each opening its
   page. Enter searches; Ctrl+Enter (or the AI button) asks for an answer. */
const suggest = $("suggest");
let picks = [];
let pick = -1;
let suggestGen = 0;

async function loadSuggest() {
  const query = box.value.trim();
  const mine = ++suggestGen;
  if (query.length < 2) return closeSuggest();
  const data = await (await fetch("/api/search?live=1&q=" + encodeURIComponent(query))).json().catch(() => null);
  if (!data || mine !== suggestGen || document.activeElement !== box) return;
  dpi = data.dpi || dpi;
  picks = data.results.slice(0, 8);
  pick = -1;
  if (!picks.length) return closeSuggest();
  suggest.innerHTML = picks.map((r, i) =>
    '<li role="option" data-pick="' + i + '"><span class="st">' + escapeHtml(r.book) + '</span>'
    + '<span class="sp">p. ' + r.page + '</span>'
    + '<span class="ss">' + markTerms(r.snippet || "", query) + '</span></li>').join("");
  suggest.hidden = false;
}

function closeSuggest() {
  suggestGen++;
  suggest.hidden = true;
  picks = [];
  pick = -1;
}

function movePick(step) {
  if (suggest.hidden || !picks.length) return;
  pick = (pick + step + picks.length + 1) % (picks.length + 1) - 1;
  [...suggest.children].forEach((li, i) => li.setAttribute("aria-selected", String(i === pick)));
}

function openPick(i) {
  const r = picks[i];
  closeSuggest();
  if (r) openBook(r.book_id, r.page, r.rects);
}

box.addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(loadSuggest, 160);
});
box.addEventListener("keydown", (event) => {
  if (event.key === "ArrowDown") { event.preventDefault(); return movePick(1); }
  if (event.key === "ArrowUp") { event.preventDefault(); return movePick(-1); }
  if (event.key === "Escape") return closeSuggest();
  if (event.key !== "Enter") return;
  event.preventDefault();
  clearTimeout(timer);
  if (pick >= 0 && !(event.ctrlKey || event.metaKey)) return openPick(pick);
  closeSuggest();
  run(event.ctrlKey || event.metaKey);
});
box.addEventListener("blur", () => setTimeout(closeSuggest, 150));
suggest.addEventListener("mousedown", (event) => {
  const li = event.target.closest("[data-pick]");
  if (li) { event.preventDefault(); openPick(Number(li.dataset.pick)); }
});
$("go").addEventListener("click", () => { clearTimeout(timer); closeSuggest(); run(false); });
$("ask").addEventListener("click", () => { clearTimeout(timer); closeSuggest(); run(true); });

// Mark the query's words in a snippet. The text is split on the terms and each
// piece escaped, so a mark never lands inside an HTML entity.
function markTerms(text, query) {
  const terms = query.toLowerCase().split(/[^a-z0-9]+/i).filter((t) => t.length > 2);
  if (!terms.length) return escapeHtml(text);
  const pattern = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  return String(text).split(new RegExp("(" + pattern + ")", "gi"))
    .map((part, i) => (i % 2 ? "<mark>" + escapeHtml(part) + "</mark>" : escapeHtml(part)))
    .join("");
}

let dpi = 130;

// The state line names what produced the results: the mode and rerank, plus AI for an answer.
// While a search runs, the current results fade under a moving bar until the new ones arrive.
const LABEL = { hybrid: "Hybrid", lexical: "Lexical", semantic: "Semantic", neural: "Neural", llm: "LLM" };
let inflight = null;
let lastAsk = null;              // the kind of the last run, so a mode change can repeat it

function took(ms) {
  return ms < 1000 ? ms + " ms" : (ms / 1000).toFixed(1) + " s";
}

function busy(on) {
  $("busy").hidden = !on;
  out.classList.toggle("pending", on);
}

async function run(ask) {
  const query = box.value.trim();
  const mine = ++generation;
  inflight?.abort();
  inflight = new AbortController();
  if (query.length < 2) return;
  lastAsk = ask;
  const label = LABEL[mode.value] + (rerank.value !== "off" ? " + " + LABEL[rerank.value] : "") + (ask ? " + AI" : "");
  busy(true);
  let res;
  try {
    res = await fetch(`/api/${ask ? "answer" : "search"}?q=${encodeURIComponent(query)}&mode=${mode.value}&rerank=${rerank.value}`,
                      { signal: inflight.signal });
  } catch (e) {
    return;                                      // aborted by a newer search
  }
  if (mine !== generation) return;
  busy(false);
  if (!res.ok) {
    state.textContent = "";
    out.className = "note";
    out.textContent = await res.text();
    return;
  }
  const data = await res.json();
  dpi = data.dpi || dpi;
  if (ask) {
    state.innerHTML = escapeHtml(label) + ", " + data.answer.quotes.length + " quotes, " + took(data.ms);
    out.className = "";
    out.innerHTML = renderAnswer(data.answer);
    return;
  }
  const seen = new Set(data.results.map((r) => r.book_id));
  state.innerHTML = escapeHtml(label) + ", " + data.results.length + " in " + seen.size + " books, " + took(data.ms);
  if (!data.results.length) {
    out.className = "note";
    out.textContent = "No matches";
    return;
  }
  out.className = "";
  out.innerHTML = '<ol class="rank">' + data.results.map((row, i) => renderRow(row, i, query)).join("") + '</ol>';
}

// The page toggle and Open button of a result or a quote. `key` names its page sheet;
// `rects` (PDF points) mark it on the page image and in the reader.
function actions(key, item) {
  const rects = escapeHtml(JSON.stringify(item.rects || []));
  return '<div class="acts">'
    + '<button class="act" data-page-for="' + key + '">' + ICON.page + 'p. ' + item.page + '</button>'
    + '<button class="act" data-open="' + item.book_id + '" data-at="' + item.page + '" data-rects="' + rects + '">'
    + ICON.book + 'Open</button>'
    + '</div>'
    + '<div class="sheet" hidden data-rects="' + rects + '" id="sheet-' + key + '">'
    + '<img alt="p. ' + item.page + '" data-src="/page?chunk=' + item.chunk_id + '">'
    + '</div>';
}

function renderRow(row, index, query) {
  const year = row.year ? '<span class="yr">' + row.year + '</span>' : "";
  const chapter = row.section ? '<div class="chapter">' + escapeHtml(row.section) + '</div>' : "";
  return '<li class="row' + (row.dropped ? ' dropped' : '') + '">'
    + '<span class="n">' + (index + 1) + '</span>'
    + '<div class="body">'
    + '<div class="head"><span class="bt">' + escapeHtml(row.book) + '</span>' + year + '</div>'
    + chapter
    + '<div class="snip">' + markTerms(row.snippet || "", query) + '</div>'
    + actions("r" + row.chunk_id, row)
    + '</div></li>';
}

// The answer is one card: the AI icon, the model's response citing the quotes by number,
// then the quotes as quotations in the books' own words, each with its source. The source
// opens the reader at the marked quote; the page icon shows the page image here.
function renderAnswer(answer) {
  if (!answer.quotes.length) return '<section class="answer"><div class="ai">' + ICON.ai + '</div><p class="note">No answer in these passages</p></section>';
  const response = answer.summary
    ? '<p class="response" title="Written by the language model">'
      + escapeHtml(answer.summary).replace(/\[(\d+)\]/g, '<button class="cite" data-cite="$1">$1</button>') + '</p>'
    : "";
  return '<section class="answer"><div class="ai">' + ICON.ai + '</div>' + response + answer.quotes.map((q) => {
    const rects = escapeHtml(JSON.stringify(q.rects || []));
    const where = escapeHtml(q.book) + (q.year ? " (" + q.year + ")" : "") + ", p. " + q.page;
    return '<figure class="q" id="quote-' + q.n + '">'
      + '<blockquote>' + escapeHtml(q.text) + '<sup>' + q.n + '</sup></blockquote>'
      + '<figcaption>'
      + '<button class="src" title="' + escapeHtml(q.section || "") + '" data-open="' + q.book_id + '" data-at="' + q.page
      + '" data-rects="' + rects + '">' + where + '</button>'
      + '<button class="ib pg" title="Page" data-page-for="q' + q.n + '">' + ICON.page + '</button>'
      + '</figcaption>'
      + '<div class="sheet" hidden data-rects="' + rects + '" id="sheet-q' + q.n + '">'
      + '<img alt="p. ' + q.page + '" data-src="/page?chunk=' + q.chunk_id + '"></div>'
      + '</figure>';
  }).join("") + '</section>';
}

// The page image loads on first toggle only.
out.addEventListener("click", (event) => {
  const cite = event.target.closest("[data-cite]");
  if (cite) return document.getElementById("quote-" + cite.dataset.cite)?.scrollIntoView({ block: "center", behavior: "smooth" });
  const opener = event.target.closest("[data-open]");
  if (opener) return openBook(opener.dataset.open, opener.dataset.at, JSON.parse(opener.dataset.rects || "[]"));
  const toggle = event.target.closest("[data-page-for]");
  if (!toggle) return;
  const sheet = document.getElementById("sheet-" + toggle.dataset.pageFor);
  const image = sheet.querySelector("img");
  if (!image.src && image.dataset.src) {
    image.addEventListener("load", () => placeBoxes(image), { once: true });
    image.src = image.dataset.src;
  }
  sheet.hidden = !sheet.hidden;
});

// Rects are in PDF points and the image is rendered at a known DPI, which gives the page size.
function placeBoxes(image) {
  const sheet = image.parentElement;
  if (sheet.querySelector(".box")) return;
  const width = image.naturalWidth * 72 / dpi;
  const height = image.naturalHeight * 72 / dpi;
  for (const [x0, y0, x1, y1] of JSON.parse(sheet.dataset.rects || "[]")) sheet.appendChild(boxAt(x0, y0, x1, y1, width, height));
}

drawTabs();
await loadBooks();
await loadLibrary();
if (!books.length) activate("library");
else await restoreTabs();
