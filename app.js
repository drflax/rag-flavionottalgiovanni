/**
 * RAG Flavio Nottalgiovanni Sambucetti — Versión estable con archivo .md
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
  'should','would','may','might','must','shall',
  'qué','quién','quiénes','cuál','cuáles','cómo','dónde','cuándo','cuánto','cuánta','cuántos','cuántas','adónde',
  'hay','ha','han','he','hemos','había','habían','tiene','tienen','tengo','tienes','tenemos','hace','hacen','hizo','hicieron'
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
  search(q, topK=12){
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
let fullMarkdown = '';

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
      .replace(/^\s*[-*+]\s+/gm, '• ')
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
    
    if(cleanText.length < 10) continue;
    
    const startsWithTitle = cleanText.toLowerCase().startsWith(title.toLowerCase());
    docs.push({
      text: startsWithTitle ? cleanText : `${title}. ${cleanText}`,
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
    if(dataEmpty) dataEmpty.textContent = 'Cargando flavio.md…';
    
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
    
    fullMarkdown = mdText;
    const docs = parseMarkdown(mdText);
    
    for(const doc of docs){
      index.add(doc);
    }
    
    mdLoaded = true;
    const st = index.stats();
    const statDocs = document.getElementById('statDocs');
    const statTerms = document.getElementById('statTerms');
    if(statDocs) statDocs.textContent = st.docs;
    if(statTerms) statTerms.textContent = st.terms;
    if(dataEmpty) dataEmpty.style.display = 'none';
    if(fileList) fileList.innerHTML = `
      <div class="file-item">
        <span class="file-name">flavio.md</span>
        <span class="file-rows">${docs.length} secciones • ${st.docs} documentos indexados</span>
      </div>
    `;
    
    return docs.length;
  }catch(err){
    console.error('Error cargando flavio.md:', err);
    if(dataEmpty) dataEmpty.innerHTML = `
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

// ---------- Extracción directa de flavio.md (fallback fiable) ----------
function sectionByNumber(md, number){
  const startRe = new RegExp(`^##\\s+${number}\\.\\s+.*$`, 'm');
  const startMatch = startRe.exec(md);
  if(!startMatch) return '';
  const start = startMatch.index;
  const rest = md.slice(start + startMatch[0].length);
  const nextRe = new RegExp(`^##\\s+${number + 1}\\.\\s+`, 'm');
  const nextMatch = nextRe.exec(rest);
  return md.slice(start, nextMatch ? start + startMatch[0].length + nextMatch.index : md.length);
}

function cleanMdLine(line){
  return line
    .replace(/^\s*[-*+]\s+/, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .trim();
}

function sectionBody(doc){
  const name = doc.meta && doc.meta.name ? doc.meta.name : '';
  let body = doc.text || '';
  if(name && body.startsWith(name)){
    body = body.slice(name.length).replace(/^[.\s]+/, '');
  }
  return body;
}

function extractCompanies(md){
  const section = sectionByNumber(md, 4);
  const lines = section.split('\n');
  const items = [];
  for(let i=0;i<lines.length;i++){
    const m = lines[i].match(/^###\s+4\.\d+\.\s+(.+)$/);
    if(!m) continue;
    const name = cleanMdLine(m[1]);
    let period = '';
    for(let j=i+1;j<Math.min(i+7, lines.length);j++){
      const p = lines[j].match(/\*\*Periodo:\*\*\s*(.+)$/);
      if(p){ period = cleanMdLine(p[1]); break; }
    }
    items.push({ name, period });
  }
  return items;
}

function extractSkills(md){
  const section = sectionByNumber(md, 3);
  const start = section.indexOf('### 3.1');
  const end = section.indexOf('### 3.2');
  const block = start >= 0 ? section.slice(start, end >= 0 ? end : section.length) : section;
  return block.split('\n')
    .filter(line => /^\s*\*\s+/.test(line))
    .map(cleanMdLine)
    .filter(Boolean);
}

function extractEducation(md){
  return sectionByNumber(md, 7).split('\n')
    .filter(line => /^\s*\*\s+/.test(line))
    .map(cleanMdLine)
    .filter(Boolean);
}

function extractField(md, label){
  const re = new RegExp(`\\*\\*${label}:\\*\\*\\s*(.+)`);
  const m = md.match(re);
  return m ? cleanMdLine(m[1]) : '';
}

function extractSectionBullets(md, number){
  return sectionByNumber(md, number).split('\n')
    .filter(line => /^###\s+/.test(line) || /^\s*\*\s+/.test(line))
    .map(line => {
      const heading = line.match(/^###\s+(.+)$/);
      return heading ? { heading: cleanMdLine(heading[1]) } : { text: cleanMdLine(line) };
    })
    .filter(item => item.heading || item.text);
}

function renderSection(items){
  let html = '';
  let inList = false;
  for(const item of items){
    if(item.heading){
      if(inList){ html += '</ul>'; inList = false; }
      html += `<div style="margin-top:.55rem"><strong>${escapeHtml(item.heading)}</strong></div>`;
    }else{
      if(!inList){ html += '<ul>'; inList = true; }
      html += `<li>${escapeHtml(item.text)}</li>`;
    }
  }
  if(inList) html += '</ul>';
  return html;
}

const QUERY_SYNONYMS = [
  [/(empresa|trabaj|experiencia|laboral|empleo|cargo|puesto|compañ|compan|trayectoria)/, 'experiencia laboral empresa trabajo desarrollador periodo responsabilidades'],
  [/(habilidad|competencia|stack|tecnolog|sabe|conocimiento|herramienta|lenguaje)/, 'habilidades competencias técnicas desarrollo php cms sapui5'],
  [/(estudi|formaci|educaci|titul|universidad|certific|curso)/, 'formación académica educación titulación certificación idep bcn'],
  [/(headline|titular|profesi|ocupaci|qué hace|que hace)/, 'titular profesional desarrollador full stack gestor deportivo'],
  [/(contact|email|correo|teléfono|telefono|móvil|movil|web|portfolio|portafolio)/, 'contacto correo electrónico teléfono sitio web portafolio'],
  [/(ubicaci|dónde vive|donde vive|localidad|ciudad|residencia|\breside\b|\bvive\b)/, 'ubicación base barcelona vilamajor']
];

function expandQuery(q){
  let out = q;
  const ql = q.toLowerCase();
  for(const [re, extra] of QUERY_SYNONYMS){
    if(re.test(ql)) out += ' ' + extra;
  }
  return out;
}

function answerLocal(q){
  const md = fullMarkdown;
  if(!md){
    return 'Todavía no se ha cargado <code>flavio.md</code>. Espera un momento y vuelve a intentarlo.';
  }

  const ql = q.toLowerCase();
  const blocks = [];

  const mentionsSports = /(deport|mountainboard|balonmano|ima\b|asociaci)/.test(ql);
  const mentionsWork = /(empresa|trabaj|laboral|empleo|cargo|puesto|compañ|compan|trayectoria|currículum|curriculum)/.test(ql);
  const mentionsExperience = /experiencia/.test(ql);

  const wantsCompanies = mentionsWork || (mentionsExperience && !mentionsSports);
  const wantsSports = mentionsSports || (mentionsExperience && !mentionsWork);
  const wantsSkills = /(habilidad|competencia|stack|tecnolog|sabe|conocimiento|herramienta|lenguaje)/.test(ql);
  const wantsEducation = /(estudi|formaci|educaci|titul|universidad|colegio|instituto|certific|curso)/.test(ql);
  const wantsHeadline = /(headline|titular|profesi|ocupaci|qué hace|que hace)/.test(ql);
  const wantsContact = /(contact|email|correo|teléfono|telefono|móvil|movil|web|portfolio|portafolio|linkedin)/.test(ql);
  const wantsLocation = /(ubicaci|dónde vive|donde vive|localidad|ciudad|residencia|\breside\b|\bvive\b)/.test(ql);
  const wantsTeaching = /(docen|profesor|docente|audiovisual|fotograf|televisi|realizaci|tecnocasa|comercial|ventas)/.test(ql);

  if(wantsCompanies){
    const companies = extractCompanies(md);
    if(companies.length){
      blocks.push(`<strong>Empresas y roles (sección 4 de flavio.md):</strong><ul>${companies.map(c=>`<li><strong>${escapeHtml(c.name)}</strong>${c.period?` — <span style="color:var(--muted)">${escapeHtml(c.period)}</span>`:''}</li>`).join('')}</ul>`);
    }
    blocks.push('También es fundador y director de <strong>El Sot Mountainboard Center</strong>, presidente de la <strong>Associació Catalana de Mountainboard</strong> y fue <strong>International Relations Manager</strong> de la <strong>IMA</strong>.');
  }

  if(wantsSports){
    const sports = extractSectionBullets(md, 5);
    if(sports.length){
      blocks.push(`<strong>Gestión deportiva y asociacionismo (sección 5):</strong>${renderSection(sports)}`);
    }
  }

  if(wantsTeaching){
    const teaching = extractSectionBullets(md, 6);
    if(teaching.length){
      blocks.push(`<strong>Docencia, audiovisual y comercial (sección 6):</strong>${renderSection(teaching)}`);
    }
  }

  if(wantsSkills){
    const skills = extractSkills(md);
    if(skills.length){
      blocks.push(`<strong>Habilidades técnicas (sección 3.1):</strong><ul>${skills.map(s=>`<li>${escapeHtml(s)}</li>`).join('')}</ul>`);
    }
  }

  if(wantsEducation){
    const education = extractEducation(md);
    if(education.length){
      blocks.push(`<strong>Formación y certificaciones (sección 7):</strong><ul>${education.map(e=>`<li>${escapeHtml(e)}</li>`).join('')}</ul>`);
    }
  }

  if(wantsHeadline){
    const headline = extractField(md, 'Titular profesional');
    if(headline) blocks.push(`<strong>Titular profesional:</strong> ${escapeHtml(headline)}`);
  }

  if(wantsContact){
    const fields = [
      { label: 'Correo electrónico', value: extractField(md, 'Correo electrónico de contacto') },
      { label: 'Teléfono', value: extractField(md, 'Teléfono de contacto') },
      { label: 'Sitio web', value: extractField(md, 'Sitio web / Portafolio') }
    ].filter(f => f.value);
    if(fields.length){
      blocks.push(`<strong>Contacto:</strong><ul>${fields.map(f=>`<li>${escapeHtml(f.label)}: ${escapeHtml(f.value)}</li>`).join('')}</ul>`);
    }
  }

  if(wantsLocation){
    const location = extractField(md, 'Ubicación base');
    if(location) blocks.push(`<strong>Ubicación:</strong> ${escapeHtml(location)}`);
  }

  if(blocks.length){
    return `<div style="line-height:1.6">${blocks.join('<div style="margin-top:.75rem"></div>')}</div>`;
  }

  const results = index.search(expandQuery(q), 8);
  const qTokens = tokenize(q);
  const relevant = results.filter(r=>{
    if(qTokens.length===0) return false;
    const docTokens = new Set(r.tokens);
    const matched = qTokens.filter(t=>docTokens.has(t)).length;
    return matched / qTokens.length >= 0.5;
  });

  if(relevant.length===0){
    return 'No encontré información sobre eso en <code>flavio.md</code>. Prueba con: <em>“¿en qué empresas ha trabajado?”</em>, <em>“¿qué habilidades tiene?”</em>, <em>“¿dónde estudió?”</em>, <em>“¿cuál es su titular profesional?”</em> o <em>“¿cómo contactarlo?”</em>.';
  }

  const list = relevant.slice(0,6).map(r=>{
    const meta = r.meta;
    const body = sectionBody(r);
    return `<div style="margin:.45rem 0;padding:.55rem;background:var(--surface);border:1px solid var(--border);border-radius:8px;"><strong>${escapeHtml(meta.name)}</strong><div style="font-size:.85rem;margin-top:.3rem;color:var(--muted)">${escapeHtml(body.substring(0,500))}${body.length>500?'…':''}</div></div>`;
  }).join('');
  return `<div style="line-height:1.5">Información relevante de <code>flavio.md</code>:${list}</div>`;
}

// ---------- Workers AI (backend Cloudflare) ----------
function formatAnswer(text){
  let safe = escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^\s*[*\-]\s+/gm, '• ');
  return safe.replace(/\n{2,}/g, '<br><br>').replace(/\n/g, '<br>');
}

async function answerWithAI(q){
  try{
    // flavio.md es pequeño (~10 KB): se envía completo para que el modelo
    // no dependa de una recuperación TF-IDF que puede fallar.
    const context = fullMarkdown;
    if(!context){
      return answerLocal(q);
    }

    const controller = new AbortController();
    const timer = setTimeout(()=>controller.abort(), 45000);
    let res;
    try{
      res = await fetch(`${API_BASE}/api/query`, {
        method:'POST',
        headers:{'content-type':'application/json'},
        body: JSON.stringify({ question: q, context }),
        signal: controller.signal
      });
    }finally{
      clearTimeout(timer);
    }

    if(!res.ok){
      const err = await res.json().catch(()=>({}));
      throw new Error(err.error || `Error ${res.status}`);
    }

    const data = await res.json();

    if(data.fallback || !data.answer){
      const note = data.refusal
        ? '📄 Sin datos sobre eso en <code>flavio.md</code>.'
        : `📄 Respuesta extraída directamente de <code>flavio.md</code>${data.error ? ` (${escapeHtml(String(data.error))})` : ''}.`;
      return answerLocal(q) + `<div style="margin-top:.7rem;font-size:.78rem;color:var(--muted)">${note}</div>`;
    }

    let html = `<div style="font-size:0.95rem; line-height:1.6;">${formatAnswer(data.answer)}</div>`;

    if(data.model){
      html += `<div style="margin-top:0.5rem;font-size:0.72rem;color:var(--muted)">Generado con ${escapeHtml(data.model)} (Workers AI) · Fuente: flavio.md</div>`;
    }

    return html;
  }catch(err){
    const message = err && err.name === 'AbortError'
      ? 'Workers AI tardó demasiado en responder'
      : String(err.message || err);
    return answerLocal(q) +
      `<div style="margin-top:.7rem;font-size:.78rem;color:var(--muted)">📄 Respuesta extraída directamente de <code>flavio.md</code> (${escapeHtml(message)}).</div>`;
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

loadMarkdown().then(count=>{
  if(count>0){
    renderBot(`Listo. He cargado los datos de <strong>Flavio Nottalgiovanni Sambucetti</strong> desde flavio.md (${count} secciones). Pregúntame lo que quieras.<br><br>Ejemplos: <em>“¿en qué empresas ha trabajado Flavio?”</em>, <em>“¿qué habilidades tiene?”</em>, <em>“¿dónde estudió?”</em>, <em>“¿cuál es su headline?”</em>`);
  } else {
    renderBot('No se pudieron cargar los datos. Revisa el mensaje de error en el panel izquierdo.');
  }
});
