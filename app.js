/**
 * RAG LinkedIn CSV — Versión específica para Flavio Nottalgiovanni Sambucetti
 * Motor TF-IDF + Workers AI para responder SOLO sobre los datos de Flavio
 */

const API_BASE = ''; // mismo origen: /api/query
const USE_WORKERS_AI = true;

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
const files = new Map(); // name -> { rows, headers }

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

function getField(row, ...keys){
  const normalizedKeys = keys.map(k => k.toLowerCase().replace(/[ _]/g,''));
  for(const rk of Object.keys(row)){
    const nk = rk.toLowerCase().replace(/[ _]/g,'');
    if(normalizedKeys.includes(nk)) return row[rk];
  }
  return '';
}

function buildDocument(row, source, type){
  let text = '', meta = {};

  switch(type){
    case 'profile': {
      const first = getField(row,'firstname')||'';
      const last = getField(row,'lastname')||'';
      const name = (first+' '+last).trim() || 'Flavio Nottalgiovanni Sambucetti';
      const headline = getField(row,'headline')||'';
      const summary = getField(row,'summary')||'';
      const industry = getField(row,'industry')||'';
      const location = getField(row,'geolocation','location')||'';
      const websites = getField(row,'websites')||'';

      text = [
        `Perfil de LinkedIn de ${name}`,
        headline ? `Headline: ${headline}` : '',
        industry ? `Industria: ${industry}` : '',
        location ? `Ubicación: ${location}` : '',
        summary ? `Resumen profesional: ${summary}` : '',
        websites ? `Webs: ${websites}` : ''
      ].filter(Boolean).join('. ');

      meta = { name, type:'profile', headline, industry, location, websites, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'positions': {
      const company = getField(row,'companyname','company')||'';
      const title = getField(row,'title')||'';
      const location = getField(row,'location')||'';
      const started = getField(row,'startedon','start date')||'';
      const finished = getField(row,'finishedon','end date')||'';
      const desc = getField(row,'description')||'';

      text = [
        `Experiencia laboral: ${title || 'Cargo'} en ${company || 'Empresa'}`,
        location ? `Ubicación: ${location}` : '',
        started ? `Desde: ${started}` : '',
        finished ? `Hasta: ${finished}` : '',
        desc ? `Descripción: ${desc}` : ''
      ].filter(Boolean).join('. ');

      meta = { name: company||'Posición', type:'position', company, title, location, started, finished, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'education': {
      const school = getField(row,'schoolname','school')||'';
      const degree = getField(row,'degree name','degree')||'';
      const start = getField(row,'start date')||'';
      const end = getField(row,'end date')||'';
      const notes = getField(row,'notes')||'';

      text = [
        `Educación: ${school || 'Institución'}`,
        degree ? `Título: ${degree}` : '',
        start ? `Desde: ${start}` : '',
        end ? `Hasta: ${end}` : '',
        notes ? `Notas: ${notes}` : ''
      ].filter(Boolean).join('. ');

      meta = { name: school||'Educación', type:'education', school, degree, start, end, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'skills': {
      const name = getField(row,'name')||'';
      text = `Habilidad: ${name}`;
      meta = { name, type:'skill', skill: name, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'certifications': {
      const name = getField(row,'name')||'';
      const authority = getField(row,'authority')||'';
      const url = getField(row,'url')||'';
      const started = getField(row,'startedon','start date')||'';
      const finished = getField(row,'finishedon','end date')||'';

      text = [
        `Certificación: ${name}`,
        authority ? `Autoridad: ${authority}` : '',
        started ? `Desde: ${started}` : '',
        finished ? `Hasta: ${finished}` : '',
        url ? `URL: ${url}` : ''
      ].filter(Boolean).join('. ');

      meta = { name, type:'certification', authority, url, started, finished, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'invitations': {
      const from = getField(row,'from')||'';
      const to = getField(row,'to')||'';
      const sent = getField(row,'sent at')||'';
      const msg = getField(row,'message')||'';
      const direction = getField(row,'direction')||'';

      text = [
        `Invitación de LinkedIn: ${direction || 'saliente'}`,
        from ? `De: ${from}` : '',
        to ? `Para: ${to}` : '',
        sent ? `Fecha: ${sent}` : '',
        msg ? `Mensaje: ${msg}` : ''
      ].filter(Boolean).join('. ');

      meta = { name: to||from||'Contacto', type:'invitation', from, to, sent, message: msg, direction, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'companyfollows': {
      const org = getField(row,'organization')||'';
      const followed = getField(row,'followed on')||'';

      text = [
        `Empresa seguida en LinkedIn: ${org}`,
        followed ? `Seguido el: ${followed}` : ''
      ].filter(Boolean).join('. ');

      meta = { name: org||'Empresa', type:'companyfollow', organization: org, followed, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'notes': {
      const first = getField(row,'connection firstname','firstname')||'';
      const last = getField(row,'connection lastname','lastname')||'';
      const url = getField(row,'connection profile url','profile url')||'';
      const note = getField(row,'note')||'';
      const created = getField(row,'created on')||'';
      const name = (first+' '+last).trim()||'Nota';

      text = [
        `Nota sobre contacto: ${name}`,
        note ? `Nota: ${note}` : '',
        created ? `Creada: ${created}` : '',
        url ? `Perfil: ${url}` : ''
      ].filter(Boolean).join('. ');

      meta = { name, type:'note', first, last, url, note, created, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'emails': {
      const email = getField(row,'email address','email')||'';
      text = `Email: ${email}`;
      meta = { name: email||'Email', type:'email', email, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'phones': {
      const phone = getField(row,'phone number','phone')||'';
      text = `Teléfono: ${phone}`;
      meta = { name: phone||'Teléfono', type:'phone', phone, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'organizations': {
      const name = getField(row,'name')||'';
      text = `Organización: ${name}`;
      meta = { name, type:'organization', source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'richmedia': {
      const title = getField(row,'title','name')||'';
      const url = getField(row,'url')||'';
      text = [
        `Rich media: ${title}`,
        url ? `URL: ${url}` : ''
      ].filter(Boolean).join('. ');
      meta = { name: title||'Rich media', type:'richmedia', url, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'whatsapp': {
      const number = getField(row,'whatsapp phone number','phone')||'';
      text = `WhatsApp: ${number}`;
      meta = { name: number||'WhatsApp', type:'whatsapp', phone: number, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'registration': {
      const name = getField(row,'name','full name')||'';
      text = `Registro: ${name}`;
      meta = { name, type:'registration', source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    case 'savedjobs': {
      const title = getField(row,'title','job title')||'';
      const company = getField(row,'company name','company')||'';
      text = [
        `Alerta de empleo: ${title}`,
        company ? `Empresa: ${company}` : ''
      ].filter(Boolean).join('. ');
      meta = { name: title||'Alerta empleo', type:'savedjob', title, company, source, persona:'Flavio Nottalgiovanni Sambucetti' };
      break;
    }
    default: {
      const pairs = [];
      for(const [k,v] of Object.entries(row)){
        if(v) pairs.push(`${k}: ${v}`);
      }
      text = pairs.join('. ');
      meta = { name: 'Documento', type:'generic', source, persona:'Flavio Nottalgiovanni Sambucetti' };
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

// ---------- Respuesta local (TF-IDF) ----------
function answerLocal(q){
  const results = index.search(q, 6);
  if(results.length===0){
    return 'No encontré información relevante sobre Flavio Nottalgiovanni Sambucetti en los CSV cargados.';
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
  const list = items.map(r=>{
    const meta = r.meta;
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
  return `Información sobre Flavio Nottalgiovanni Sambucetti:<br>${list}<div style="margin-top:.6rem;font-size:.78rem;color:var(--muted)">Relevancia: ${scores}</div>`;
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

    if(data.fallback){
      return renderLocalResults(data.results, q) +
        `<div style="margin-top:.6rem;font-size:.78rem;color:var(--danger)">⚠️ Workers AI no disponible (${escapeHtml(String(data.error||''))}). Mostrando resultados locales.</div>`;
    }

    let html = `<div style="font-size:0.95rem; line-height:1.6;">${escapeHtml(data.answer||'')}</div>`;

    if(data.results && data.results.length){
      html += `<div style="margin-top:1rem; border-top:1px solid var(--border); padding-top:0.75rem;">
        <div style="font-size:0.78rem; color:var(--muted); margin-bottom:0.5rem;">📚 Fuentes de Flavio:</div>
        ${data.results.slice(0,5).map(r=>{
          const meta = r.meta || r;
          const chips = [];
          if(meta.title) chips.push(`<em>${escapeHtml(meta.title)}</em>`);
          if(meta.company) chips.push(`en <strong>${escapeHtml(meta.company)}</strong>`);
          return `<div style="margin:.3rem 0;padding:.4rem;background:var(--surface);border:1px solid var(--border);border-radius:6px;font-size:0.88rem;">
            <strong>${escapeHtml(meta.name)}</strong> ${chips.join(' ')}
            <span style="font-size:.72rem;opacity:.6">• ${Math.round(r.score*100)}% • ${escapeHtml(meta.type)} • ${escapeHtml(meta.source)}</span>
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

function renderLocalResults(results, q){
  if(!results || results.length===0){
    return 'No encontré información relevante sobre Flavio Nottalgiovanni Sambucetti.';
  }
  const seen = new Set();
  const items = [];
  for(const r of results){
    const meta = r.meta || r;
    const key = meta.name + '|' + meta.type + '|' + meta.source;
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
  return `Información sobre Flavio Nottalgiovanni Sambucetti:<br>${list}<div style="margin-top:.6rem;font-size:.78rem;color:var(--muted)">Relevancia: ${scores}</div>`;
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
  renderBot('Datos limpiados. Sube los CSV de LinkedIn de Flavio Nottalgiovanni Sambucetti.');
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
renderBot('Listo. Sube los archivos CSV de LinkedIn de <strong>Flavio Nottalgiovanni Sambucetti</strong> y empieza a preguntar. <br><br>Ejemplos: <em>“¿en qué empresas ha trabajado Flavio?”</em>, <em>“¿qué habilidades tiene?”</em>, <em>“¿dónde estudió?”</em>, <em>“¿cuál es su headline?”</em>');
