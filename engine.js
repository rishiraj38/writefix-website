/* WriteFix in the browser: the same network as wf/model.py and the same safety logic as wf/infer.py,
   written in plain JavaScript so the demo runs on the visitor's own device with nothing sent anywhere. */
var WF = (function () {
  const PAD = 0, BOS = 1, EOS = 2, FIX = 4, GATE = 0.85, CHUNK = 45;
  let T = {}, cfg = null, vocab = null, ivocab = [], ranks = null, b2c = [], c2b = {}, bpeCache = new Map();
  const RE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  function halfToFloat(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
      out[i] = e === 0 ? s * 6.103515625e-5 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return out;
  }

  async function load(get, onProgress) {
    const td = new TextDecoder();
    const man = JSON.parse(td.decode(await get('model/manifest.json')));
    cfg = man.cfg;
    const bufs = [];
    for (let i = 0; i < man.files.length; i++) { const b64 = td.decode(await get('model/' + man.files[i])), bin = atob(b64), u8 = new Uint8Array(bin.length); for (let k = 0; k < bin.length; k++) u8[k] = bin.charCodeAt(k); bufs.push(new Uint16Array(u8.buffer)); if (onProgress) onProgress((i + 1) / (man.files.length + 1)); }
    for (const t of man.tensors) T[t.name] = halfToFloat(bufs[t.file].subarray(t.offset, t.offset + t.len));
    const tok = JSON.parse(td.decode(await get('model/tokenizer.json')));
    vocab = tok.model.vocab; ivocab = []; for (const k in vocab) ivocab[vocab[k]] = k;
    ranks = new Map(); tok.model.merges.forEach((m, i) => ranks.set(Array.isArray(m) ? m[0] + ' ' + m[1] : m, i));
    const bs = []; for (let b = 33; b <= 126; b++) bs.push(b); for (let b = 161; b <= 172; b++) bs.push(b); for (let b = 174; b <= 255; b++) bs.push(b);
    const cs = bs.slice(); let n = 0; for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    bs.forEach((b, i) => { b2c[b] = String.fromCharCode(cs[i]); c2b[b2c[b]] = b; });
    if (onProgress) onProgress(1);
    return man;
  }

  // ---------- tokenizer (byte-level BPE) ----------
  function bpe(word) {
    if (bpeCache.has(word)) return bpeCache.get(word);
    let parts = [...word];
    while (parts.length > 1) {
      let best = Infinity, at = -1;
      for (let i = 0; i < parts.length - 1; i++) { const r = ranks.get(parts[i] + ' ' + parts[i + 1]); if (r !== undefined && r < best) { best = r; at = i; } }
      if (at < 0) break;
      const a = parts[at], b = parts[at + 1], merged = [];
      for (let i = 0; i < parts.length; i++) { if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { merged.push(a + b); i++; } else merged.push(parts[i]); }
      parts = merged;
    }
    bpeCache.set(word, parts); return parts;
  }
  function encode(text) {
    if (!text.startsWith(' ')) text = ' ' + text;
    const ids = [], enc = new TextEncoder();
    for (const m of text.match(RE) || []) {
      const w = [...enc.encode(m)].map(b => b2c[b]).join('');
      for (const p of bpe(w)) ids.push(vocab[p] !== undefined ? vocab[p] : 3);
    }
    return ids;
  }
  function decode(ids) {
    const bytes = []; for (const id of ids) for (const ch of ivocab[id] || '') if (c2b[ch] !== undefined) bytes.push(c2b[ch]);
    return new TextDecoder().decode(new Uint8Array(bytes));
  }

  // ---------- maths ----------
  function linear(x, n, W, b, nin, nout) {
    const y = new Float32Array(n * nout);
    for (let t = 0; t < n; t++) { const xo = t * nin, yo = t * nout;
      for (let o = 0; o < nout; o++) { let s = b ? b[o] : 0; const wo = o * nin; for (let i = 0; i < nin; i++) s += W[wo + i] * x[xo + i]; y[yo + o] = s; } }
    return y;
  }
  function lnorm(x, n, d, g, b) {
    const y = new Float32Array(n * d);
    for (let t = 0; t < n; t++) { const o = t * d; let m = 0; for (let i = 0; i < d; i++) m += x[o + i]; m /= d;
      let v = 0; for (let i = 0; i < d; i++) { const z = x[o + i] - m; v += z * z; } const s = 1 / Math.sqrt(v / d + 1e-5);
      for (let i = 0; i < d; i++) y[o + i] = (x[o + i] - m) * s * g[i] + b[i]; }
    return y;
  }
  function erf(x) { const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); }
  function gelu(x) { for (let i = 0; i < x.length; i++) x[i] = 0.5 * x[i] * (1 + erf(x[i] * 0.7071067811865476)); return x; }
  function add(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; return a; }
  /* q: nq x d, k/v: nk x d (already projected). Returns nq x d before the output projection. */
  function attend(q, nq, k, v, nk, d, H, causal) {
    const hd = d / H, out = new Float32Array(nq * d), sc = new Float32Array(nk), scale = 1 / Math.sqrt(hd);
    for (let h = 0; h < H; h++) { const ho = h * hd;
      for (let i = 0; i < nq; i++) { const lim = causal ? i + 1 : nk; let mx = -Infinity;
        for (let j = 0; j < lim; j++) { let s = 0; const qo = i * d + ho, ko = j * d + ho; for (let c = 0; c < hd; c++) s += q[qo + c] * k[ko + c]; s *= scale; sc[j] = s; if (s > mx) mx = s; }
        let z = 0; for (let j = 0; j < lim; j++) { sc[j] = Math.exp(sc[j] - mx); z += sc[j]; }
        const oo = i * d + ho; for (let j = 0; j < lim; j++) { const w = sc[j] / z, vo = j * d + ho; for (let c = 0; c < hd; c++) out[oo + c] += w * v[vo + c]; } } }
    return out;
  }
  const L = (p, x, n, nin, nout) => linear(x, n, T[p + '.weight'], T[p + '.bias'], nin, nout);
  function ff(p, x, n, d, f) { return L(p + '.b', gelu(L(p + '.a', x, n, d, f)), n, f, d); }

  function encodeSrc(ids) {
    const d = cfg.d, n = ids.length, E = T['emb.weight'], P = T['pos_e.weight'], sq = Math.sqrt(d);
    let x = new Float32Array(n * d);
    for (let t = 0; t < n; t++) for (let i = 0; i < d; i++) x[t * d + i] = E[ids[t] * d + i] * sq + P[t * d + i];
    for (let s = 0; s < 2; s++) { const p = 'stem.' + s, h = lnorm(x, n, d, T[p + '.norm.weight'], T[p + '.norm.bias']);
      const w = T[p + '.depth.weight'], b = T[p + '.depth.bias'], c1 = new Float32Array(n * d);
      for (let t = 0; t < n; t++) for (let c = 0; c < d; c++) { let a = b[c]; for (let j = 0; j < 5; j++) { const tt = t + j - 2; if (tt >= 0 && tt < n) a += w[c * 5 + j] * h[tt * d + c]; } c1[t * d + c] = a; }
      add(x, linear(gelu(c1), n, T[p + '.point.weight'], T[p + '.point.bias'], d, d)); }
    for (let l = 0; l < cfg.enc; l++) { const p = 'enc.' + l, h = lnorm(x, n, d, T[p + '.n1.weight'], T[p + '.n1.bias']);
      const a = attend(L(p + '.att.q', h, n, d, d), n, L(p + '.att.k', h, n, d, d), L(p + '.att.v', h, n, d, d), n, d, cfg.heads, false);
      add(x, L(p + '.att.o', a, n, d, d));
      add(x, ff(p + '.ff', lnorm(x, n, d, T[p + '.n2.weight'], T[p + '.n2.bias']), n, d, cfg.ff)); }
    return lnorm(x, n, d, T['ne.weight'], T['ne.bias']);
  }

  /* Greedy decoding with a key/value cache. onStep lets the page breathe and show progress. */
  async function generate(src, onStep) {
    const d = cfg.d, n = src.length, mem = encodeSrc(src), E = T['emb.weight'], P = T['pos_d.weight'], sq = Math.sqrt(d), V = cfg.vocab, ob = T['out_bias'];
    const layers = []; for (let l = 0; l < cfg.dec; l++) { const p = 'dec.' + l; layers.push({ p, sk: [], sv: [], ck: L(p + '.cross.k', mem, n, d, d), cv: L(p + '.cross.v', mem, n, d, d) }); }
    const maxNew = Math.min(cfg.max_len - 2, Math.floor(n * 1.3) + 12), ids = [], probs = [];
    let tok = BOS, lpSum = 0;
    for (let t = 0; t < maxNew; t++) {
      let x = new Float32Array(d); for (let i = 0; i < d; i++) x[i] = E[tok * d + i] * sq + P[t * d + i];
      for (const ly of layers) { const p = ly.p;
        let h = lnorm(x, 1, d, T[p + '.n1.weight'], T[p + '.n1.bias']);
        ly.sk.push(L(p + '.self_att.k', h, 1, d, d)); ly.sv.push(L(p + '.self_att.v', h, 1, d, d));
        const m = ly.sk.length, K = new Float32Array(m * d), Vv = new Float32Array(m * d); for (let j = 0; j < m; j++) { K.set(ly.sk[j], j * d); Vv.set(ly.sv[j], j * d); }
        add(x, L(p + '.self_att.o', attend(L(p + '.self_att.q', h, 1, d, d), 1, K, Vv, m, d, cfg.heads, false), 1, d, d));
        h = lnorm(x, 1, d, T[p + '.n2.weight'], T[p + '.n2.bias']);
        add(x, L(p + '.cross.o', attend(L(p + '.cross.q', h, 1, d, d), 1, ly.ck, ly.cv, n, d, cfg.heads, false), 1, d, d));
        add(x, ff(p + '.ff', lnorm(x, 1, d, T[p + '.n3.weight'], T[p + '.n3.bias']), 1, d, cfg.ff)); }
      const h = lnorm(x, 1, d, T['nd.weight'], T['nd.bias']);
      let best = -Infinity, bi = 0; const lg = new Float32Array(V);
      for (let o = 0; o < V; o++) { let s = ob[o]; const wo = o * d; for (let i = 0; i < d; i++) s += E[wo + i] * h[i]; lg[o] = s; if (s > best) { best = s; bi = o; } }
      let z = 0; for (let o = 0; o < V; o++) z += Math.exp(lg[o] - best);
      const p = 1 / z; lpSum += Math.log(p);
      if (bi === EOS) break;
      ids.push(bi); probs.push(p); tok = bi;
      if (onStep && t % 3 === 0) await onStep(ids);
    }
    return { ids, probs, lp: lpSum / (ids.length + 1) };
  }

  // ---------- difflib.SequenceMatcher (autojunk off), needed to match the Python safety gate exactly ----------
  function blocks(a, b) {
    const b2j = new Map(); b.forEach((x, i) => { if (!b2j.has(x)) b2j.set(x, []); b2j.get(x).push(i); });
    function longest(alo, ahi, blo, bhi) { let bi = alo, bj = blo, bs = 0, j2 = new Map();
      for (let i = alo; i < ahi; i++) { const nj = new Map();
        for (const j of (b2j.get(a[i]) || [])) { if (j < blo) continue; if (j >= bhi) break; const k = (j2.get(j - 1) || 0) + 1; nj.set(j, k); if (k > bs) { bi = i - k + 1; bj = j - k + 1; bs = k; } }
        j2 = nj; }
      return [bi, bj, bs]; }
    const q = [[0, a.length, 0, b.length]], res = [];
    while (q.length) { const [alo, ahi, blo, bhi] = q.pop(), [i, j, k] = longest(alo, ahi, blo, bhi);
      if (k) { res.push([i, j, k]); if (alo < i && blo < j) q.push([alo, i, blo, j]); if (i + k < ahi && j + k < bhi) q.push([i + k, ahi, j + k, bhi]); } }
    res.sort((x, y) => x[0] - y[0] || x[1] - y[1]); res.push([a.length, b.length, 0]); return res;
  }
  function opcodes(a, b) { let i = 0, j = 0; const out = [];
    for (const [ai, bj, n] of blocks(a, b)) { const tag = i < ai && j < bj ? 'replace' : i < ai ? 'delete' : j < bj ? 'insert' : '';
      if (tag) out.push([tag, i, ai, j, bj]); i = ai + n; j = bj + n; if (n) out.push(['equal', ai, i, bj, j]); }
    return out; }
  function ratio(a, b) { const m = blocks([...a], [...b]).reduce((s, x) => s + x[2], 0); return a.length + b.length ? 2 * m / (a.length + b.length) : 1; }
  const key = w => w.toLowerCase().replace(/[^a-z0-9]/g, '');

  function gate(src, words) {
    const sw = src.replace(/\s+([.,?!;:]+)(?=\s|$)/g, '$1').split(/\s+/).filter(Boolean), hw = words.map(w => w[0]), res = [];
    function keep(seg, like) { seg.forEach((w, k) => {
      if (like && k === seg.length - 1) w = w.replace(/[.,?!;:]+$/, '') + (like.match(/[.,?!;:]*$/) || [''])[0];
      if (like && k === 0 && /^[A-Z]/.test(like) && (!res.length || /[.?!]$/.test(res[res.length - 1]))) w = w.charAt(0).toUpperCase() + w.slice(1);
      res.push(w); }); }
    for (const [op, i1, i2, j1, j2] of opcodes(sw.map(key), hw.map(key))) {
      if (op === 'equal') for (let j = j1; j < j2; j++) res.push(hw[j]);
      else if (op === 'replace' && i2 - i1 === j2 - j1) for (let i = i1, j = j1; i < i2; i++, j++) { if (words[j][1] >= GATE) res.push(hw[j]); else keep([sw[i]], hw[j]); }
      else if (op === 'replace') { let mn = 1; for (let j = j1; j < j2; j++) mn = Math.min(mn, words[j][1]); if (mn >= GATE) for (let j = j1; j < j2; j++) res.push(hw[j]); else keep(sw.slice(i1, i2), hw[j2 - 1]); }
      else if (op === 'insert') { for (let j = j1; j < j2; j++) if (words[j][1] >= GATE) res.push(hw[j]); }
      else for (let i = i1; i < i2; i++) if (!(i > 0 && key(sw[i]) === key(sw[i - 1]))) res.push(sw[i]);
    }
    return res.join(' ');
  }

  async function raw(text, onStep) {
    const src = [FIX].concat(encode(text).slice(0, 300), [EOS]);
    const { ids, probs, lp } = await generate(src, onStep ? (p) => onStep(decode(p).trim()) : null);
    const words = [];
    ids.forEach((t, i) => { const piece = ivocab[t]; if (piece.startsWith('Ġ') || !words.length) words.push([[t], 1]); else words[words.length - 1][0].push(t);
      if (/[a-z0-9]/i.test(piece)) words[words.length - 1][1] = Math.min(words[words.length - 1][1], probs[i]); });
    const ws = words.map(w => [decode(w[0]).trim(), w[1]]).filter(w => w[0]);
    return { hyp: gate(text, ws), lp };
  }
  function safe(src, hyp, lp) {
    if (!hyp) return false;
    const r = hyp.length / Math.max(src.length, 1), w = hyp.split(/\s+/).length / Math.max(src.split(/\s+/).length, 1);
    if (!(r >= 0.7 && r <= 1.5 && w >= 0.75 && w <= 1.4 && lp > -0.9)) return false;
    const k = t => t.toLowerCase().replace(/[^a-z0-9 ]/g, '');
    return ratio(k(src), k(hyp)) >= 0.7;
  }
  async function fixLine(line, onStep) {
    const text = line.replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/[ \t]+/g, ' ').trim();
    if (!text || /[^\x00-\x7f]/.test(text) || text.split(' ').length < 2) return line;
    const words = text.split(' '), done = []; let carry = '';
    for (let i = 0; i < words.length;) {
      const src = (carry + ' ' + words.slice(i, i + CHUNK).join(' ')).trim(); i += CHUNK;
      const pre = done.join(' ');
      let { hyp, lp } = await raw(src, onStep ? (p) => onStep((pre + ' ' + p).trim()) : null);
      if (!safe(src, hyp, lp)) hyp = src;
      carry = '';
      if (i < words.length) { const cut = Math.max(hyp.lastIndexOf('. '), hyp.lastIndexOf('? '), hyp.lastIndexOf('! '));
        if (cut > hyp.length * 0.4) { carry = hyp.slice(cut + 2); hyp = hyp.slice(0, cut + 1); }
        else if (/[.?!]$/.test(hyp) && !/[.?!]$/.test(src)) hyp = hyp.slice(0, -1); }
      done.push(hyp);
    }
    return done.filter(Boolean).join(' ');
  }
  async function fix(text, onStep) { const out = []; for (const l of text.split('\n')) out.push(await fixLine(l, onStep ? (p) => onStep(out.concat([p]).join('\n')) : null)); return out.join('\n'); }

  return { load, fix, encode, decode, opcodes: (a, b) => opcodes(a.map(key), b.map(key)).map(o => o), exact: (a, b) => opcodes(a, b), ready: () => !!cfg };
})();
if (typeof module !== 'undefined') module.exports = WF;
