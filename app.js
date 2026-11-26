/**
 * RAG Flavio Nottalgiovanni Sambucetti — Versión con archivo .md
 * Lee flavio.md, construye índice TF-IDF local, opcional Workers AI para respuesta generativa.
 */

const API_BASE = ''; // mismo origen: /api/query
const USE_WORKERS_AI = true;
const MD_URL = 'flavio.md';

// ---------- Stopwords ----------
const STOPWORDS = new Set([
  'el','la','los','las','un','una','unos','unas','y','o','pero','por','para','con','sin','sobre',
  'de','del','a','en','que','quien','quienes','cual','cuales','cuyo','cuya','cuyos','cuyas',
  'este','esta','estos','estas','ese','esa','esos','esas','aquel','aquella','aquellos','aquellas',
  'lo','le','se','me','te','nos','os','mi','mis','tu','tus','su','sus','nuestro','nuestra','nuestros','nuestras',
  'ser','estar','tener','haber','hacer','ir','decir','ver','querer','poder','deber','saber','parecer',
  'es','son','está','están','estoy','estás','estamos','estáis','están','fui','fuiste','fue','fuimos','fuisteis','fueron',
  'era','eras','éramos','eran','seré','serás','será','seremos','seréis','serán',
  'más','menos','muy','bien','mal','siempre','nunca','ya','todavía','aún','ahora','antes','después','entonces',
  'también','tampoco','sino','aunque','porque','puesto','ya que','como','si','o bien','ni','tanto','cuanto',
  'the','a','an','and','or','but','if','then','else','when','at','from','by','for','with','about','against',
  'between','into','through','during','before','after','above','below','to','of','in','on','off','over','under',
  'again','further','once','here','there','when','where','why','how','all','any','both','each','few','more',
  'most','other','some','such','no','nor','not','only','own','same','so','than','too','very','just','don',
  'now','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing',
  'i','me','my','myself','we','our','ours','you','your','yours','he','him','his','she','her','hers','it','its',
  'they','them','their','theirs','what','which','who','whom','this','that','these','those','am','can','could',
  'should','would','may','might','must','shall'
]);

function tokenize(text){
  if(!text) return [];
  return String(text)
    .toLowerCase()
    .replace(/[^a-záéíóúñü0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 1 && !STOPWORDS.has(t));
}

// ---------- Motor TF-IDF ----------
class RagIndex {
  constructor(){ this.docs=[]; this.df=new Map(); this.idf=new Map(); this.vectors=[]; }
  add(doc){
    const id = this.docs.length;
    const tokens = tokenize(doc.text);
    const tf = new Map();
    for(const t of tokens) tf.set(t, (tf.get(t)||0)+1);
    let maxF = 1;
    for(const v of tf.values()) if(v>maxF) maxF=v;
    const vec = new Map();
    for(const [t,c] of tf) vec.set(t, (0.5 + 0.5*(c/maxF)));
    this.docs.push({ id, text: doc.text, meta: doc.meta||{}, source: doc.source||'', tokens, vec });
    for(const t of vec.keys()) this.df.set(t, (this.df.get(t)||0)+1);
    this.recomputeIdf();
    this.vectors[id] = vec;
  }
  recomputeIdf(){
    const N = Math.max(1, this.docs.length);
    for(const [t,df] of this.df) this.idf.set(t, Math.log(N/df) + 1);
  }
  queryVector(tokens){
    const tf = new Map();
    for(const t of tokens) tf.set(t, (tf.get(t)||0)+1);
    let maxF=1; for(const v of tf.values()) if(v>maxF) maxF=v;
    const vec = new Map();
    for(const [t,c] of tf){
      const idf = this.idf.get(t);
      if(!idf) continue;
      vec.set(t, (0.5+0.5*(c/maxF)) * idf);
    }
    return vec;
  }
  cosine(a,b){
    if(a.size===0 || b.size===0) return 0;
    let dot=0, na=0, nb=0;
    const smaller = a.size<=b.size? a:b;
    const larger = a.size<=b.size? b:a;
    for(const [t,va] of smaller){
      const vb = larger.get(t);
      if(vb){ dot += va*vb; na += va*va; } else { na += va*va; }
    }
    for(const vb of larger.values()) nb += vb*vb;
    const denom = Math.sqrt(na)*Math.sqrt(nb);
    return denom===0?0:dot/denom;
  }
  search(q, topK=10){
    const qTokens = tokenize(q);
    const qVec = this.queryVector(qTokens);
    const scored = [];
    for(const doc of this.docs){
      const s = this.cosine(qVec, doc.vec);
      if(s>0) scored.push({ score: s, ...doc });
    }
    scored.sort((a,b)=> b.score-a.score);
    return scored.slice(0, topK);
  }
  stats(){
    const terms = new Set();
    for(const d of this.docs) for(const t of d.vec.keys()) terms.add(t);
    return { docs: this.docs.length, terms: terms.size };
  }
}

const index = new RagIndex();
let mdLoaded = false;

// ---------- Parseador de Markdown ----------
function parseMarkdown(mdText){
  const docs = [];
  // Dividir por encabezados ## o #
  const sections = mdText.split(/(?=^#{1,3}\s)/m);
  
  for(const section of sections){
    const trimmed = section.trim();
    if(!trimmed) continue;
    
    const titleMatch = trimmed.match(/^#{1,3}\s+(.+)$/m);
    const title = titleMatch ? titleMatch[1].trim() : 'Sin título';
    
    let cleanText = trimmed
      .replace(/^#{1,3}\s+/gm, '')
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/^\s*[-*+]\s+/gm, '• ')
      .replace(/\n{3,}/g, '\n\n');
    
    if(cleanText.trim().length < 10) continue;
    
    docs.push({
      text: `${title}. ${cleanText}`,
      meta: {
        name: title,
        type: 'markdown-section',
        source: 'flavio.md',
        persona: 'Flavio Nottalgiovanni Sambucetti'
      }
    });
  }
  
  if(docs.length === 0){
    docs.push({
      text: mdText,
      meta: { name: 'Flavio Nottalgiovanni Sambucetti - Perfil completo', type: 'markdown', source: 'flavio.md', persona: 'Flavio Nottalgiovanni Sambucetti' }
    });
  }
  
  return docs;
}

async function loadMarkdown(){
  const dataEmpty = document.getElementById('dataEmpty');
  const fileList = document.getElementById('fileList');
  
  try{
    dataEmpty.textContent = 'Cargando flavio.md…';
    
    const res = await fetch(MD_URL, { cache: 'no-cache' });
    
    if(res.status === 404){
      throw new Error('Archivo flavio.md no encontrado en el servidor (404).');
    }
    if(!res.ok){
      throw new Error(`Error HTTP ${res.status} al cargar flavio.md`);
    }
    
    const mdText = await res.text();
    
    if(!mdText || mdText.trim().length === 0){
      throw new Error('El archivo flavio.md está vacío.');
    }
    
    const docs = parseMarkdown(mdText);
    
    for(const doc of docs){
      index.add(doc);
    }
    
    mdLoaded = true;
    const st = index.stats();
    document.getElementById('statDocs').textContent = st.docs;
    document.getElementById('statTerms').textContent = st.terms;
    dataEmpty.style.display = 'none';
    fileList.innerHTML = `
      <div class="file-item">
        <span class="file-name">flavio.md</span>
        <span class="file-rows">${docs.length} secciones • ${st.docs} documentos indexados</span>
      </div>
    `;
    
    return docs.length;
  }catch(err){
    console.error('Error cargando flavio.md:', err);
    dataEmpty.innerHTML = `
      <div style="color: var(--danger); margin-bottom: 1rem;">
        ❌ Error cargando flavio.md: ${escapeHtml(err.message)}
      </div>
      <div style="font-size: 0.85rem; color: var(--muted);">
        <strong>Posibles soluciones:</strong><br>
        1. Verifica que <code>flavio.md</code> exista en la raíz del repositorio<br>
        2. Vuelve a desplegar en Cloudflare Pages<br>
        3. Abre <code>flavio.md</code> directamente en el navegador para confirmar que es accesible
      </div>
    `;
    return 0;
  }
}

// ---------- Respuesta local (TF-IDF) ----------
function answerLocal(q){
  const results = index.search(q, 15);
  if(results.length===0){
    return 'No encontré información relevante sobre Flavio Nottalgiovanni Sambucetti en el archivo flavio.md.<br><br>Intenta preguntar de forma más específica, por ejemplo: <em>“¿en qué empresas ha trabajado Flavio?”</em>, <em>“¿qué habilidades tiene?”</em>, <em>“¿dónde estudió?”</em>';
  }
  const seen = new Set();
  const items = [];
  for(const r of results){
    const key = r.meta.name + '|' + r.meta.source;
    if(seen.has(key)) continue;
    seen.add(key);
    items.push(r);
    if(items.length>=6) break;
  }
  const list = items.map(r=>{
    const meta = r.meta;
    let html = `<strong>${escapeHtml(meta.name)}</strong>`;
    html += ` <span style="font-size:.78rem;opacity:.6">• ${Math.round(r.score*100)}% • ${escapeHtml(meta.source)}</span>`;
    return `<div style="margin:.45rem 0;padding:.5rem;background:var(--surface);border:1px solid var(--border);border-radius:8px;">${html}<div style="font-size:.85rem;margin-top:.25rem;color:var(--muted)">${escapeHtml(r.text.substring(0,300))}${r.text.length>300?'…':''}</div></div>`;
  }).join('');
  const scores = results.slice(0,3).map(r=>Math.round(r.score*100)).join(', ');
  return `Información sobre Flavio Nottalgiovanni Sambucetti:<br>${list}<div style="margin-top:.6rem;font-size:.78rem;color:var(--muted)">Relevancia: ${scores}</div>`;
}

// ---------- Workers AI (backend Cloudflare) ----------
async function answerWithAI(q){
  try{
    const results = index.search(q, 15);
    const context = results.length
      ? results.map(r => `- ${r.meta.name}\n  ${r.text}`).join('\n\n')
      : 'No hay contexto disponible.';

    const res = await fetch(`${API_BASE}/api/query`, {
      method:'POST',
      headers:{'content-type':'application/json'},
      body: JSON.stringify({ question: q, context })
    });
    if(!res.ok){
      const err = await res.json().catch(()=>({}));
      throw new Error(err.error || `Error ${res.status}`);
    }
    const data = await res.json();

    if(data.fallback){
      return answerLocal(q) +
        `<div style="margin-top:.6rem;font-size:.78rem;color:var(--danger)">⚠️ Workers AI no disponible (${escapeHtml(String(data.error||''))}). Mostrando resultados locales.</div>`;
    }

    let html = `<div style="font-size:0.95rem; line-height:1.6;">${escapeHtml(data.answer||'')}</div>`;

    if(data.results && data.results.length){
      html += `<div style="margin-top:1rem; border-top:1px solid var(--border); padding-top:0.75rem;">
        <div style="font-size:0.78rem; color:var(--muted); margin-bottom:0.5rem;">📚 Fuentes de Flavio:</div>
        ${data.results.slice(0,6).map(r=>{
          const meta = r.meta || r;
          return `<div style="margin:.3rem 0;padding:.4rem;background:var(--surface);border:1px solid var(--border);border-radius:6px;font-size:0.88rem;">
            <strong>${escapeHtml(meta.name)}</strong>
            <span style="font-size:.72rem;opacity:.6">• ${Math.round(r.score*100)}% • ${escapeHtml(meta.source)}</span>
          </div>`;
        }).join('')}
      </div>`;
    }

    if(data.model){
      html += `<div style="margin-top:0.5rem;font-size:0.72rem;color:var(--muted)">Generado con ${data.model} (Workers AI)</div>`;
    }

    return html;
  }catch(err){
    return answerLocal(q) + `<div style="margin-top:.6rem;font-size:.78rem;color:var(--danger)">⚠️ Backend no disponible (${escapeHtml(String(err.message))}). Modo local activado.</div>`;
  }
}

// ---------- UI ----------
const msgEl = document.getElementById('messages');
function scrollBottom(){ msgEl.scrollTop = msgEl.scrollHeight; }
function setLoading(b){
  const btn = document.getElementById('btnAsk');
  const input = document.getElementById('question');
  btn.disabled = b; input.disabled = b;
  btn.textContent = b ? 'Procesando con IA…' : 'Preguntar';
}
function hint(msg){ document.getElementById('hint').textContent = msg||''; }
function renderBot(html){
  const t = document.getElementById('tplMsg');
  const el = t.content.cloneNode(true);
  el.querySelector('.msg').classList.add('msg-bot');
  el.querySelector('.bubble').innerHTML = html;
  msgEl.appendChild(el);
  scrollBottom();
}
function renderUser(text){
  const t = document.getElementById('tplMsg');
  const el = t.content.cloneNode(true);
  el.querySelector('.msg').classList.add('msg-you');
  el.querySelector('.bubble').textContent = text;
  msgEl.appendChild(el);
  scrollBottom();
}
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c=> ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[c])); }

// ---------- Eventos ----------
document.getElementById('composer').onsubmit = async (e)=>{
  e.preventDefault();
  const input = document.getElementById('question');
  const q = input.value.trim();
  if(!q) return;
  if(!mdLoaded){
    renderBot('Los datos de Flavio aún no se han cargado. Espera a que termine la carga de flavio.md y vuelve a intentarlo.');
    return;
  }
  renderUser(q);
  input.value='';
  setLoading(true);
  await new Promise(r=>setTimeout(r, 120));
  try{
    const html = USE_WORKERS_AI ? await answerWithAI(q) : answerLocal(q);
    renderBot(html);
  }catch(err){
    renderBot('Error: '+escapeHtml(String(err)));
  }finally{
    setLoading(false);
    input.focus();
  }
};

// Init


loadMarkdown().then(count=>{
  if(count>0){
    renderBot(`Listo. He cargado los datos de <strong>Flavio Nottalgiovanni Sambucetti</strong> desde flavio.md (${count} secciones). Pregúntame lo que quieras.<br><br>Ejemplos: <em>“¿en qué empresas ha trabajado Flavio?”</em>, <em>“¿qué habilidades tiene?”</em>, <em>“¿dónde estudió?”</em>, <em>“¿cuál es su headline?”</em>`);
  } else {
    renderBot('No se pudieron cargar los datos. Revisa el mensaje de error en el panel izquierdo.');
  }
});
