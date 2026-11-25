/**
 * Cloudflare Pages Function — POST /api/ingest
 * Recibe un archivo CSV (multipart/form-data, campo "file"), lo parsea y guarda el índice en KV.
 *
 * Body (form-data): file = archivo CSV
 * Respuesta: { ok: true, file, rows, added, totalDocs }
 */

import { RagIndex, parseCsv, csvTypeFromName, buildDocument, serializeIndex } from '../_rag/index.js';

export const onRequestPost = async ({ env, request }) => {
  try {
    const formData = await request.formData();
    const file = formData.get('file');
    if (!file || !(file instanceof File)) {
      return new Response(JSON.stringify({ error: 'Falta archivo "file"' }), {
        status: 400,
        headers: { 'content-type': 'application/json' }
      });
    }

    const text = await file.text();
    const { headers, rows } = parseCsv(text);
    if (rows.length === 0) {
      return new Response(JSON.stringify({ error: 'CSV sin filas de datos' }), {
        status: 400,
        headers: { 'content-type': 'application/json' }
      });
    }

    const index = await loadIndex(env.LINKEDIN_KV);
    const type = csvTypeFromName(file.name);
    let added = 0;

    for (const row of rows) {
      const doc = buildDocument(row, file.name, type);
      if (doc.text.trim()) { index.add(doc); added++; }
    }

    if (env.LINKEDIN_KV) {
      await env.LINKEDIN_KV.put('linkedin:index', serializeIndex(index));
    }

    return new Response(JSON.stringify({
      ok: true,
      file: file.name,
      type,
      rows: rows.length,
      added,
      totalDocs: index.stats().docs
    }), {
      headers: { 'content-type': 'application/json' }
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: err.message }), {
      status: 500,
      headers: { 'content-type': 'application/json' }
    });
  }
};

async function loadIndex(kv) {
  const index = new RagIndex();
  if (!kv) return index;
  const raw = await kv.get('linkedin:index');
  if (!raw) return index;
  try {
    const data = JSON.parse(raw);
    for (const d of data.docs || []) index.add(d);
  } catch (e) {
    // índice corrupto, empezar vacío
  }
  return index;
}
