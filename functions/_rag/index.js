// Código compartido entre el frontend y las funciones de Cloudflare Pages
// Motor TF-IDF + parseadores de CSV de LinkedIn

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

export function tokenize(text){
  if(!text) return [];
  return String(text)
    .toLowerCase()
    .replace(/[^a-záéíóúñü0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 1 && !STOPWORDS.has(t));
}

export class RagIndex {
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

export function parseCsv(text){
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

export function parseLine(line, sep){
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

export function csvTypeFromName(name){
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

export function buildDocument(row, source, type){
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

export function serializeIndex(index){
  return JSON.stringify({ docs: index.docs.map(d => ({ id: d.id, text: d.text, meta: d.meta, source: d.source, tokens: d.tokens })) });
}

export function deserializeIndex(data){
  const index = new RagIndex();
  for (const d of data.docs || []) index.add(d);
  return index;
}
