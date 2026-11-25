/**
 * RAG LinkedIn CSV — Motor de búsqueda + Workers AI (Cloudflare)
 * - Modo local: TF-IDF + coseno en el navegador (sin claves de API)
 * - Modo Cloudflare: envía a /api/query (búsqueda + Workers AI para respuesta generativa)
 *
 * Soporta los CSV de LinkedIn Data Export:
 *   Profile, Positions, Education, Skills, Certifications, Invitations,
 *   Company Follows, Notes, Email Addresses, PhoneNumbers, Organizations,
 *   Rich_Media, Whatsapp Phone Numbers, Registration, SavedJobAlerts
 */

const API_BASE = ''; // mismo origen: /api/query y /api/ingest
const USE_WORKERS_AI = true; // si true, intenta usar el backend de Cloudflare Pages

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

// ---------- Motor TF-IDF local ----------
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
  search(q, topK=8){
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
const files = new Map(); // name -> { rows, headers, type }

// ---------- Parseadores ----------
function parseCsv(text){
  const lines = text.split(/\r?\n/).filter(l => l.trim().length>0);
  if(lines.length===0) return { headers:[], rows:[] };
  const sep = lines[0].includes(';') ? ';' : ',';
  const headers = parseLine(lines[0], sep).map(h=>h.trim().replace(/"/g,''));
  const rows = [];
  for(let i=1;i<lines.length;i++){
    const vals = parseLine(lines[i], sep);
    const obj = {};
    headers.forEach((h,idx)=> obj[h] = (vals[idx]||'').replace(/"/g,'').trim());
    rows.push(obj);
  }
  return { headers, rows };
}
function parseLine(line, sep){
  const out=[]; let cur=''; let inQ=false;
  for(let i=0;i<line.length;i++){
    const ch=line[i];
    if(ch==='"'){ inQ=!inQ; continue; }
    if(ch===sep && !inQ){ out.push(cur); cur=''; continue; }
    cur+=ch;
  }
  out.push(cur);
  return out;
}

function csvTypeFromName(name){
  const n = name.toLowerCase();
  if(n.includes('profile')) return 'profile';
  if(n.includes('position')) return 'positions';
  if(n.includes('education')) return 'education';
  if(n.includes('skill')) return 'skills';
  if(n.includes('certification')) return 'certifications';
  if(n.includes('invitation')) return 'invitations';
  if(n.includes('company follow') || n.includes('companyfollow')) return 'companyfollows';
  if(n.includes('note')) return 'notes';
  if(n.includes('email')) return 'emails';
  if(n.includes('phone')) return 'phones';
  if(n.includes('organization')) return 'organizations';
  if(n.includes('rich media') || n.includes('rich_media')) return 'richmedia';
  if(n.includes('whatsapp')) return 'whatsapp';
  if(n.includes('registration')) return 'registration';
  if(n.includes('saved job') || n.includes('savedjob')) return 'savedjobs';
  return 'generic';
}

function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c=> ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[c])); }

function buildDocument(row, source, type){
  const get = (...keys) => {
    for(const k of keys){
      for(const rk of Object.keys(row)){
        if(rk.toLowerCase().replace(/[ _]/g,'') === k.toLowerCase().replace(/[ _]/g,'')){
          return row[rk];
        }
      }
    }
    return '';
  };

  let text = '', meta = {};

  switch(type){
    case 'profile': {
      const first = get('firstname')||'';
      const last = get('lastname')||'';
      const name = (first+' '+last).trim() || 'Perfil';
      const headline = get('headline');
      const summary = get('summary');
      const industry = get('industry');
      const location = get('geolocation')||get('location')||'';
      const websites = get('websites');
      text = [
        name ? `Perfil de LinkedIn: ${name}` : '',
        headline ? `Headline: ${headline}` : '',
        industry ? `Industria: ${industry}` : '',
        location ? `Ubicación: ${location}` : '',
        summary ? `Resumen: ${summary}` : '',
        websites ? `Webs: ${websites}` : ''
      ].filter(Boolean).join('. ');
      meta = { name, type:'profile', headline, industry, location, websites, source };
      break;
    }
    case 'positions': {
      const company = get('companyname')||get('company');
      const title = get('title');
      const location = get('location');
      const started = get('startedon')||get('start date');
      const finished = get('finishedon')||get('end date');
      const desc = get('description');
      text = [
        company ? `Empresa: ${company}` : '',
        title ? `Cargo: ${title}` : '',
        location ? `Ubicación: ${location}` : '',
        started ? `Desde: ${started}` : '',
        finished ? `Hasta: ${finished}` : '',
        desc ? `Descripción: ${desc}` : ''
      ].filter(Boolean).join('. ');
      meta = { name: company||'Posición', type:'position', company, title, location, started, finished, source };
      break;
    }
    case 'education': {
      const school = get('schoolname')||get('school');
      const degree = get('degree name')||get('degree');
      const start = get('start date');
      const end = get('end date');
      const notes = get('notes');
      text = [
        school ? `Institución: ${school}` : '',
        degree ? `Título: ${degree}` : '',
        start ? `Desde: ${start}` : '',
        end ? `Hasta: ${end}` : '',
        notes ? `Notas: ${notes}` : ''
      ].filter(Boolean).join('. ');
      meta = { name: school||'Educación', type:'education', school, degree, start, end, source };
      break;
    }
    case 'skills': {
      const name = get('name');
      text = `Habilidad: ${name}`;
      meta = { name, type:'skill', skill: name, source };
      break;
    }
    case 'certifications': {
      const name = get('name');
      const authority = get('authority');
      const url = get('url');
      const started = get('startedon')||get('start date');
      const finished = get('finishedon')||get('end date');
      text = [
        name ? `Certificación: ${name}` : '',
        authority ? `Autoridad: ${authority}` : '',
        started ? `Desde: ${started}` : '',
        finished ? `Hasta: ${finished}` : '',
        url ? `URL: ${url}` : ''
      ].filter(Boolean).join('. ');
      meta = { name, type:'certification', authority, url, started, finished, source };
      break;
    }
    case 'invitations': {
      const from = get('from');
      const to = get('to');
      const sent = get('sent at');
      const msg = get('message');
      const direction = get('direction');
      const name = to || from || 'Contacto';
      text = [
        direction ? `Dirección: ${direction}` : '',
        from ? `De: ${from}` : '',
        to ? `Para: ${to}` : '',
        sent ? `Enviado: ${sent}` : '',
        msg ? `Mensaje: ${msg}` : ''
      ].filter(Boolean).join('. ');
      meta = { name, type:'invitation', from, to, sent, message: msg, direction, source };
      break;
    }
    case 'companyfollows': {
      const org = get('organization');
      const followed = get('followed on');
      text = [
        org ? `Sigue a la empresa: ${org}` : '',
        followed ? `Seguido el: ${followed}` : ''
      ].filter(Boolean).join('. ');
      meta = { name: org||'Empresa seguida', type:'companyfollow', organization: org, followed, source };
      break;
    }
    case 'notes': {
      const first = get('connection firstname')||get('firstname');
      const last = get('connection lastname')||get('lastname');
      const url = get('connection profile url')||get('profile url');
      const note = get('note');
      const created = get('created on');
      const name = (first+' '+last).trim()||'Nota';
      text = [
        name ? `Nota sobre: ${name}` : '',
        note ? `Nota: ${note}` : '',
        created ? `Creada: ${created}` : '',
        url ? `Perfil: ${url}` : ''
      ].filter(Boolean).join('. ');
      meta = { name, type:'note', first, last, url, note, created, source };
      break;
    }
    case 'emails': {
      const email = get('email address')||get('email');
      text = `Email: ${email}`;
      meta = { name: email||'Email', type:'email', email, source };
      break;
    }
    case 'phones': {
      const phone = get('phone number')||get('phone');
      text = `Teléfono: ${phone}`;
      meta = { name: phone||'Teléfono', type:'phone', phone, source };
      break;
    }
    case 'organizations': {
      const name = get('name');
      text = `Organización: ${name}`;
      meta = { name, type:'organization', source };
      break;
    }
    case 'richmedia': {
      const title = get('title')||get('name');
      const url = get('url');
      text = [
        title ? `Rich media: ${title}` : '',
        url ? `URL: ${url}` : ''
      ].filter(Boolean).join('. ');
      meta = { name: title||'Rich media', type:'richmedia', url, source };
      break;
    }
    case 'whatsapp': {
      const number = get('whatsapp phone number')||get('phone');
      text = `WhatsApp: ${number}`;
      meta = { name: number||'WhatsApp', type:'whatsapp', phone: number, source };
      break;
    }
    case 'registration': {
      const name = get('name')||get('full name');
      text = `Registro: ${name}`;
      meta = { name, type:'registration', source };
      break;
    }
    case 'savedjobs': {
      const title = get('title')||get('job title');
      const company = get('company name')||get('company');
      text = [
        title ? `Alerta de empleo: ${title}` : '',
        company ? `Empresa: ${company}` : ''
      ].filter(Boolean).join('. ');
      meta = { name: title||'Alerta empleo', type:'savedjob', title, company, source };
      break;
    }
    default: {
      const pairs = [];
      for(const [k,v] of Object.entries(row)){
        if(v) pairs.push(`${k}: ${v}`);
      }
      text = pairs.join('. ');
      meta = { name: row['Nombre']||row['Name']||'Documento', type:'generic', source };
    }
  }

  return { text: text.trim(), meta };
}

function ingestFile(name, text){
  const { headers, rows } = parseCsv(text);
  if(rows.length===0) return { ok:false, error:'El CSV no tiene filas de datos.' };
  const type = csvTypeFromName(name);
  let added=0;
  for(const row of rows){
    const doc = buildDocument(row, name, type);
    if(doc.text){ index.add(doc); added++; }
  }
  files.set(name, { rows, headers, type, count: rows.length });
  return { ok:true, added, headers, rows: rows.length, type };
}

// ---------- Intenciones ----------
function detectIntent(q){
  const ql = q.toLowerCase();
  const mCompany = ql.match(/en\s+([a-záéíóúñü0-9\s&]+?)(?:\?|$)/);
  const mRole = ql.match(/cargo\s+(?:de\s+)?([a-záéíóúñü\s]+)/) || ql.match(/trabaj[aá]?\s+como\s+([a-záéíóúñü\s]+)/);
  return { ql, mCompany, mRole };
}

// ---------- Respuesta local (TF-IDF) ----------
function answerLocal(q){
  const results = index.search(q, 6);
  if(results.length===0){
    return 'No encontré información relevante en tus CSV cargados. Prueba con otras palabras o asegúrate de haber subido los archivos de LinkedIn.';
  }
  const seen = new Set();
  const items = [];
  for(const r of results){
    const key = r.meta.name + '|' + r.meta.type + '|' + r.meta.source;
    if(seen.has(key)) continue;
    seen.add(key);
    items.push(r);
    if(items.length>=5) break;
  }
  const intent = detectIntent(q);
  let intro = '';
  if(intent.mCompany){
    const c = intent.mCompany[1].trim();
    const hits = items.filter(r => (r.meta.company||r.meta.organization||'').toLowerCase().includes(c));
    intro = hits.length ? `En <strong>${escapeHtml(c)}</strong> encontré: ` : `No encontré contactos con <strong>${escapeHtml(c)}</strong>. Resultados más parecidos: `;
  } else if(intent.mRole){
    intro = `Contactos con cargo relacionado: `;
  } else {
    intro = `Según tus archivos de LinkedIn, los resultados más relevantes son: `;
  }
  const list = items.map(r=>{
    const chips = [];
    if(r.meta.title) chips.push(`<em>${escapeHtml(r.meta.title)}</em>`);
    if(r.meta.company) chips.push(`en <strong>${escapeHtml(r.meta.company)}</strong>`);
    if(r.meta.organization) chips.push(`empresa: <strong>${escapeHtml(r.meta.organization)}</strong>`);
    if(r.meta.skill) chips.push(`habilidad: <strong>${escapeHtml(r.meta.skill)}</strong>`);
    let html = `<strong>${escapeHtml(r.meta.name)}</strong> ${chips.join(' ')}`;
    if(r.meta.url) html += ` <a href="${escapeHtml(r.meta.url)}" target="_blank" rel="noopener">ver perfil</a>`;
    html += ` <span style="font-size:.78rem;opacity:.6">• ${Math.round(r.score*100)}% • ${escapeHtml(r.meta.type)} • ${escapeHtml(r.meta.source)}</span>`;
    return `<div style="margin:.45rem 0;padding:.5rem;background:var(--surface);border:1px solid var(--border);border-radius:8px;">${html}</div>`;
  }).join('');
  const scores = results.slice(0,3).map(r=>Math.round(r.score*100)).join(', ');
  return `${intro}${list}<div style="margin-top:.6rem;font-size:.78rem;color:var(--muted)">Relevancia: ${scores}</div>`;
}

// ---------- Workers AI (backend Cloudflare) ----------
async function answerWithAI(q){
  try{
    const res = await fetch(`${API_BASE}/api/query`, {
      method:'POST',
      headers:{'content-type':'application/json'},
      body: JSON.stringify({ question: q })
    });
    if(!res.ok){
      const err = await res.json().catch(()=>({}));
      throw new Error(err.error || `Error ${res.status}`);
    }
    const data = await res.json();

    // Si Workers AI falló, el backend devuelve fallback:true con resultados locales
    if(data.fallback){
      return renderLocalResults(data.results, q) +
        `<div style="margin-top:.6rem;font-size:.78rem;color:var(--danger)">⚠️ Workers AI no disponible (${escapeHtml(String(data.error||''))}). Mostrando resultados locales.</div>`;
    }

    // Respuesta generativa de Workers AI
    let html = `<div style="font-size:0.95rem; line-height:1.6;">${escapeHtml(data.answer||'')}</div>`;

    if(data.results && data.results.length){
      html += `<div style="margin-top:1rem; border-top:1px solid var(--border); padding-top:0.75rem;">
        <div style="font-size:0.78rem; color:var(--muted); margin-bottom:0.5rem;">📚 Fuentes relevantes:</div>
        ${data.results.slice(0,5).map(r=>{
          const chips = [];
          if(r.title) chips.push(`<em>${escapeHtml(r.title)}</em>`);
          if(r.company) chips.push(`en <strong>${escapeHtml(r.company)}</strong>`);
          return `<div style="margin:.3rem 0;padding:.4rem;background:var(--surface);border:1px solid var(--border);border-radius:6px;font-size:0.88rem;">
            <strong>${escapeHtml(r.name)}</strong> ${chips.join(' ')}
            <span style="font-size:.72rem;opacity:.6">• ${Math.round(r.score*100)}% • ${escapeHtml(r.type)} • ${escapeHtml(r.source)}</span>
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

// Render de resultados locales (mismo formato que answerLocal pero aceptando array de resultados)
function renderLocalResults(results, q){
  if(!results || results.length===0){
    return 'No encontré información relevante en tus CSV cargados.';
  }
  const seen = new Set();
  const items = [];
  for(const r of results){
    const key = r.meta ? (r.meta.name + '|' + r.meta.type + '|' + r.meta.source) : (r.name+'|'+r.type+'|'+r.source);
    if(seen.has(key)) continue;
    seen.add(key);
    items.push(r);
    if(items.length>=5) break;
  }
  const list = items.map(r=>{
    const meta = r.meta || r;
    const chips = [];
    if(meta.title) chips.push(`<em>${escapeHtml(meta.title)}</em>`);
    if(meta.company) chips.push(`en <strong>${escapeHtml(meta.company)}</strong>`);
    if(meta.organization) chips.push(`empresa: <strong>${escapeHtml(meta.organization)}</strong>`);
    if(meta.skill) chips.push(`habilidad: <strong>${escapeHtml(meta.skill)}</strong>`);
    let html = `<strong>${escapeHtml(meta.name)}</strong> ${chips.join(' ')}`;
    if(meta.url) html += ` <a href="${escapeHtml(meta.url)}" target="_blank" rel="noopener">ver perfil</a>`;
    html += ` <span style="font-size:.78rem;opacity:.6">• ${Math.round(r.score*100)}% • ${escapeHtml(meta.type)} • ${escapeHtml(meta.source)}</span>`;
    return `<div style="margin:.45rem 0;padding:.5rem;background:var(--surface);border:1px solid var(--border);border-radius:8px;">${html}</div>`;
  }).join('');
  const scores = results.slice(0,3).map(r=>Math.round(r.score*100)).join(', ');
  return `Según tus archivos de LinkedIn, los resultados más relevantes son:<br>${list}<div style="margin-top:.6rem;font-size:.78rem;color:var(--muted)">Relevancia: ${scores}</div>`;
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
function renderFileList(){
  const container = document.getElementById('fileList');
  const empty = document.getElementById('dataEmpty');
  const t = document.getElementById('tplFile');
  container.innerHTML='';
  if(files.size===0){ empty.style.display='block'; } else { empty.style.display='none'; }
  for(const [name, f] of files){
    const el = t.content.cloneNode(true);
    el.querySelector('.file-name').textContent = name;
    el.querySelector('.file-rows').textContent = `${f.rows} filas (${f.type})`;
    el.querySelector('.file-remove').onclick = ()=> removeFile(name);
    container.appendChild(el);
  }
  const st = index.stats();
  document.getElementById('statDocs').textContent = st.docs;
  document.getElementById('statTerms').textContent = st.terms;
}
function removeFile(name){
  files.delete(name);
  index.docs=[]; index.df=new Map(); index.idf=new Map(); index.vectors=[];
  renderFileList();
  hint('Índice reconstruido. Vuelve a subir los CSV que quieras conservar.');
}

async function readFile(file){
  return new Promise((res,rej)=>{
    const r = new FileReader();
    r.onload = ()=>res(r.result);
    r.onerror = rej;
    r.readAsText(file);
  });
}

document.getElementById('btnOpenData').onclick = async ()=>{
  const input = document.createElement('input');
  input.type='file'; input.multiple=true; input.accept='.csv,.txt';
  input.onchange = async (e)=>{
    const arr = Array.from(e.target.files||[]);
    if(arr.length===0) return;
    for(const file of arr){
      const text = await readFile(file);
      const res = ingestFile(file.name, text);
      if(!res.ok){ alert('Error en '+file.name+': '+res.error); continue; }
    }
    renderFileList();
    hint(`Cargados ${files.size} archivo(s). Índice: ${index.stats().docs} documentos, ${index.stats().terms} términos.`);
  };
  input.click();
};

document.getElementById('btnClear').onclick = ()=>{
  if(!confirm('¿Eliminar todos los datos cargados?')) return;
  index.docs=[]; index.df=new Map(); index.idf=new Map(); index.vectors=[]; files.clear();
  renderFileList();
  msgEl.innerHTML='';
  renderBot('Datos limpiados. Sube tus CSV de LinkedIn cuando quieras.');
  hint('');
};

document.getElementById('composer').onsubmit = async (e)=>{
  e.preventDefault();
  const input = document.getElementById('question');
  const q = input.value.trim();
  if(!q) return;
  if(index.docs.length===0){
    renderBot('Primero carga al menos un CSV de LinkedIn.');
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
renderFileList();
renderBot('Listo. Sube tus archivos CSV de LinkedIn (de la carpeta LinkedInDataExport) y empieza a preguntar. <br><br>Ejemplos: <em>“¿en qué empresas ha trabajado Flavio?”</em>, <em>“¿qué habilidades tiene?”</em>, <em>“¿dónde estudió?”</em>');
