// Blunt Hall model comparison page. Data: data/compare/compare.json + plan_*.png (scripts/pipeline/compare/c05_web.py).
// The plan images are north-up at 5 cm (red channel = wall on a measured wall, green = no measured wall nearby);
// the viewer turns the floor so its long axis runs across the screen.
(async function () {
  const $ = (id) => document.getElementById(id);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const D = await (await fetch('data/compare/compare.json')).json();
  const S = D.scores, G = D.grid, pct = (v) => `${Math.round(v * 100)} %`;
  const best = D.sweep.reduce((a, b) => (b.f_score > a.f_score ? b : a));
  const sec = (s) => (s == null ? '—' : s < 90 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

  // ---------- hero numbers
  $('heroes').innerHTML = [
    ['ours', 'Ours · LiDAR block model', 'geometry fitted to the scan', true],
    [best.run, 'SceneScript · best run', `${best.tile_m} m tiles, ${best.max_points ? best.max_points / 1000 + 'k points' : 'all points'}`],
    ['cloud2bim', 'Cloud2BIM', 'density + morphology → IFC'],
  ].map(([k, t, s, win]) => `<div class="hero${win ? ' win' : ''}"><b>F ${S[k].f_score.toFixed(2)}</b><span>${t}</span><small>${s} · completeness ${pct(S[k].completeness)}, precision ${pct(S[k].precision)}</small></div>`).join('');

  // ---------- plan viewer
  const imgs = {}, load = (src) => new Promise((ok, err) => { const i = new Image(); i.onload = () => ok(i); i.onerror = err; i.src = src; });
  imgs.ref = await load('data/compare/plan_ref.png');
  for (const m of D.show) imgs[m] = await load(`data/compare/plan_${m}.png`);
  const tinted = {};
  function tint() {                        // recolour the channel-coded images with the theme's ink / red / grey
    const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const ink = hex(css('--ink')), bad = hex(css('--bad')), ref = hex(css('--ref'));
    for (const [k, im] of Object.entries(imgs)) {
      const c = document.createElement('canvas'); c.width = im.width; c.height = im.height; const x = c.getContext('2d');
      x.drawImage(im, 0, 0); const d = x.getImageData(0, 0, c.width, c.height), a = d.data;
      for (let i = 0; i < a.length; i += 4) {
        if (k === 'ref') { const v = a[i + 3]; a[i] = ref[0]; a[i + 1] = ref[1]; a[i + 2] = ref[2]; a[i + 3] = v ? 110 : 0; continue; }
        const g = a[i] > 127, b = a[i + 1] > 127, col = g ? ink : bad;
        a[i] = col[0]; a[i + 1] = col[1]; a[i + 2] = col[2]; a[i + 3] = g || b ? 255 : 0;
      }
      x.putImageData(d, 0, 0); tinted[k] = c;
    }
  }
  let cur = D.show[0], view = null;
  const wrap = $('planwrap'), cv = $('plan'), ctx = cv.getContext('2d');
  // screen mapping: building y (long axis) -> screen x, building x -> screen y (down). image px (i, j): x = lo.x + i*res, y = lo.y + (h-1-j)*res
  const toScreen = (bx, by) => [view.ox + (by - G.lo[1]) * view.s, view.oy + (bx - G.lo[0]) * view.s];
  const toBuilding = (sx, sy) => [G.lo[0] + (sy - view.oy) / view.s, G.lo[1] + (sx - view.ox) / view.s];
  function fit() {
    const W = wrap.clientWidth, H = wrap.clientHeight, lw = G.h * G.res, lh = G.w * G.res, s = Math.min((W - 24) / lw, (H - 24) / lh);
    view = { s, ox: (W - lw * s) / 2, oy: (H - lh * s) / 2 };
  }
  function draw() {
    const W = wrap.clientWidth, H = wrap.clientHeight, r = devicePixelRatio || 1;
    if (cv.width !== W * r || cv.height !== H * r) { cv.width = W * r; cv.height = H * r; }
    ctx.setTransform(r, 0, 0, r, 0, 0); ctx.clearRect(0, 0, W, H); ctx.imageSmoothingEnabled = view.s * G.res < 2;
    // image -> screen: image column i (x east) maps to screen y, image row j (north up, so y = lo.y + (h-1-j)*res) maps to screen x
    const put = (c) => { ctx.save(); ctx.translate(view.ox + (G.h) * G.res * view.s, view.oy); ctx.rotate(Math.PI / 2); ctx.scale(G.res * view.s, G.res * view.s); ctx.drawImage(c, 0, 0); ctx.restore(); };
    put(tinted.ref); put(tinted[cur]);
    ctx.lineWidth = 1.6; ctx.strokeStyle = css('--door');
    for (const d of D.doors[cur] || []) {
      const [x0, y0] = toScreen(d[0], d[1]), [x1, y1] = toScreen(d[2], d[3]);
      ctx.strokeRect(Math.min(x0, x1) - 2, Math.min(y0, y1) - 2, Math.abs(x1 - x0) + 4, Math.abs(y1 - y0) + 4);
    }
    ctx.setLineDash([3, 3]); ctx.strokeStyle = css('--muted'); ctx.lineWidth = 1.2;
    for (const k of D.known_doors) { const [x, y] = toScreen(k[0], k[1]); ctx.beginPath(); ctx.arc(x, y, Math.max(5, 0.6 * view.s), 0, 2 * Math.PI); ctx.stroke(); }
    ctx.setLineDash([]);
    // north arrow + scale bar
    ctx.fillStyle = css('--muted'); ctx.font = '11px "IBM Plex Mono", monospace';
    const L = [5, 10, 20, 50].find((m) => m * view.s > 60) || 50, bx = 14, by = H - 16;
    ctx.fillRect(bx, by, L * view.s, 2); ctx.fillText(`${L} m`, bx, by - 6);
    ctx.fillText('N →', W - 44, H - 12);
    const s = S[cur];
    $('vstat').innerHTML = `completeness <b>${pct(s.completeness)}</b> · precision <b>${pct(s.precision)}</b> · F <b>${s.f_score.toFixed(2)}</b> · doors <b>${s.doors ?? 0}</b> · known doors found <b>${D.door_check[cur]}/8</b>`;
  }
  const tabs = $('tabs');
  tabs.insertAdjacentHTML('afterbegin', D.show.map((m) => `<button class="tab" role="tab" data-m="${m}" aria-selected="${m === cur}">${D.label[m]}<small>F ${S[m].f_score.toFixed(2)}</small></button>`).join(''));
  tabs.querySelectorAll('.tab').forEach((b) => (b.onclick = () => { cur = b.dataset.m; tabs.querySelectorAll('.tab').forEach((t) => t.setAttribute('aria-selected', t === b)); draw(); }));
  const zoom = (f, cx = wrap.clientWidth / 2, cy = wrap.clientHeight / 2) => { view.ox = cx - (cx - view.ox) * f; view.oy = cy - (cy - view.oy) * f; view.s *= f; draw(); };
  $('zin').onclick = () => zoom(1.5); $('zout').onclick = () => zoom(1 / 1.5); $('zfit').onclick = () => { fit(); draw(); };
  wrap.addEventListener('wheel', (e) => { e.preventDefault(); const r = wrap.getBoundingClientRect(); zoom(e.deltaY < 0 ? 1.2 : 1 / 1.2, e.clientX - r.left, e.clientY - r.top); }, { passive: false });
  const pts = new Map(); let pinch = null;
  wrap.addEventListener('pointerdown', (e) => { wrap.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); });
  wrap.addEventListener('pointermove', (e) => {
    const r = wrap.getBoundingClientRect(), tip = $('ptip');
    if (!pts.has(e.pointerId)) {
      const [bx, by] = toBuilding(e.clientX - r.left, e.clientY - r.top);
      tip.hidden = false; tip.textContent = `x ${bx.toFixed(1)} m · y ${by.toFixed(1)} m`; tip.style.left = `${e.clientX - r.left + 12}px`; tip.style.top = `${e.clientY - r.top + 12}px`; return;
    }
    const p = pts.get(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2) {
      const [a, b] = [...pts.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch) zoom(d / pinch, (a[0] + b[0]) / 2 - r.left, (a[1] + b[1]) / 2 - r.top); pinch = d; return;
    }
    view.ox += e.clientX - p[0]; view.oy += e.clientY - p[1]; draw();
  });
  const up = (e) => { pts.delete(e.pointerId); if (pts.size < 2) pinch = null; };
  wrap.addEventListener('pointerup', up); wrap.addEventListener('pointercancel', up); wrap.addEventListener('pointerleave', () => ($('ptip').hidden = true));
  new ResizeObserver(() => { fit(); draw(); }).observe(wrap);
  tint(); fit(); draw();

  // ---------- scores table (headline methods + every SceneScript run in the sweep table below)
  const bar = (v) => `<div class="bar"><span>${pct(v)}</span><i><b style="width:${(v * 100).toFixed(1)}%"></b></i></div>`;
  const rows = [['ours', 'CPU'], ['cloud2bim', 'CPU, 32 GB RAM'], ['scenescript', 'A100 GPU'], [best.run, 'A100 GPU']];
  $('scoretable').innerHTML = `<tr><th>Method</th><th>Completeness</th><th>Precision</th><th>F</th><th>Wall length</th><th>Doors</th><th>Known doors</th><th>Run time</th></tr>`
    + rows.map(([k, hw]) => { const s = S[k]; return `<tr class="${k === 'ours' ? 'best' : ''}"><td><b>${D.label[k] || k}</b><br><span class="tag">${hw}</span></td><td>${bar(s.completeness)}</td><td>${bar(s.precision)}</td><td>${bar(s.f_score)}</td><td class="num">${Math.round(s.wall_length_m)} m</td><td class="num">${s.doors ?? 0}</td><td class="num">${D.door_check[k]} / 8</td><td class="num">${sec(s.seconds)}</td></tr>`; }).join('');

  // ---------- SceneScript tile-size sweep: F against tile size, with ours and Cloud2BIM as reference lines
  const box = $('sweepchart'), Wc = 760, Hc = 300, m = { l: 44, r: 128, t: 28, b: 40 };
  const xs = (t) => m.l + (t - 6) / (30 - 6) * (Wc - m.l - m.r), ys = (f) => Hc - m.b - f * (Hc - m.t - m.b);
  const byT = {}; for (const r of D.sweep) (byT[r.tile_m] ??= []).push(r);
  const trend = Object.keys(byT).map(Number).sort((a, b) => a - b).map((t) => [t, byT[t].reduce((a, r) => a + r.f_score, 0) / byT[t].length]);
  let svg = `<svg viewBox="0 0 ${Wc} ${Hc}" role="img" aria-label="SceneScript F-score against tile size">`;
  for (const f of [0, 0.25, 0.5, 0.75, 1]) svg += `<line class="ax" x1="${m.l}" x2="${Wc - m.r}" y1="${ys(f)}" y2="${ys(f)}"/><text class="axt" x="${m.l - 8}" y="${ys(f) + 4}" text-anchor="end">${f.toFixed(2)}</text>`;
  for (const t of [8, 12, 16, 20, 24, 28]) svg += `<text class="axt" x="${xs(t)}" y="${Hc - m.b + 18}" text-anchor="middle">${t} m</text>`;
  svg += `<text class="axt" x="${(m.l + Wc - m.r) / 2}" y="${Hc - 4}" text-anchor="middle">tile size (square, metres)</text><text class="axt" x="${m.l - 8}" y="12" text-anchor="end">F-score</text>`;
  for (const [k, cls, txt] of [['ours', 'ours', 'Ours'], ['cloud2bim', 'c2b', 'Cloud2BIM']]) {
    const y = ys(S[k].f_score); svg += `<line class="refl ${cls}" x1="${m.l}" x2="${Wc - m.r}" y1="${y}" y2="${y}"/><text class="reft" x="${Wc - m.r + 8}" y="${y + 4}">${txt} ${S[k].f_score.toFixed(2)}</text>`;
  }
  svg += `<polyline class="trend" points="${trend.map(([t, f]) => `${xs(t)},${ys(f)}`).join(' ')}"/>`;
  const last = trend[trend.length - 1]; svg += `<text class="reft" x="${Wc - m.r + 8}" y="${ys(last[1]) + 4}">SceneScript</text><text class="axt" x="${Wc - m.r + 8}" y="${ys(last[1]) + 18}">mean per tile size</text>`;
  D.sweep.forEach((r, i) => {
    const jit = (byT[r.tile_m].indexOf(r) - (byT[r.tile_m].length - 1) / 2) * 7;
    svg += `<circle class="pt" data-i="${i}" cx="${xs(r.tile_m) + jit}" cy="${ys(r.f_score)}" r="5.5"/><circle data-i="${i}" cx="${xs(r.tile_m) + jit}" cy="${ys(r.f_score)}" r="14" fill="transparent"/>`;
  });
  box.insertAdjacentHTML('beforeend', svg + '</svg>');
  const tip = $('ctip');
  box.querySelectorAll('circle[data-i]').forEach((c) => {
    c.addEventListener('pointerenter', () => {
      const r = D.sweep[+c.dataset.i]; box.querySelectorAll('.pt').forEach((p) => p.classList.toggle('hot', p.dataset.i === c.dataset.i));
      tip.innerHTML = `<b>${r.tile_m} m tiles</b>, stride ${r.stride_m} m · ${r.max_points ? r.max_points / 1000 + 'k points' : 'all points'}<br>F <b>${r.f_score.toFixed(2)}</b> · completeness ${pct(r.completeness)} · precision ${pct(r.precision)}<br>${r.tiles} tiles (${r.failed} failed, ${r.empty} empty) · ${sec(r.seconds)}`;
      const bb = box.getBoundingClientRect(), cb = c.getBoundingClientRect(); tip.hidden = false;
      tip.style.left = `${Math.min(bb.width - 260, cb.left - bb.left + 16)}px`; tip.style.top = `${cb.top - bb.top - 10}px`;
    });
    c.addEventListener('pointerleave', () => { tip.hidden = true; box.querySelectorAll('.pt').forEach((p) => p.classList.remove('hot')); });
  });
  $('sweeptable').innerHTML = `<tr><th>Tile</th><th>Stride</th><th>Points per tile</th><th>Tiles</th><th>Failed / empty</th><th>Completeness</th><th>Precision</th><th>F</th><th>Doors</th><th>Run time</th></tr>`
    + D.sweep.map((r) => `<tr class="${r === best ? 'best' : ''}"><td class="num">${r.tile_m} m</td><td class="num">${r.stride_m} m</td><td class="num">${r.max_points ? r.max_points.toLocaleString() : 'all (5 cm grid)'}</td><td class="num">${r.tiles}</td><td class="num">${r.failed} / ${r.empty}</td><td class="num">${pct(r.completeness)}</td><td class="num">${pct(r.precision)}</td><td class="num">${r.f_score.toFixed(2)}</td><td class="num">${r.doors}</td><td class="num">${sec(r.seconds)}</td></tr>`).join('');

  // ---------- theme
  $('theme').onclick = () => {
    const dark = matchMedia('(prefers-color-scheme: dark)').matches, t = document.documentElement.dataset.theme;
    document.documentElement.dataset.theme = (t ? t === 'dark' : dark) ? 'light' : 'dark'; tint(); draw();
  };
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { tint(); draw(); });
})().catch((e) => { document.querySelector('main').insertAdjacentHTML('afterbegin', `<p class="note">Could not load the comparison data: ${e.message}</p>`); console.error(e); });
