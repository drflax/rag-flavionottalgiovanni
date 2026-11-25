/**
 * Cloudflare Pages Function — POST /api/query
 * Búsqueda en el índice (KV si está disponible) + respuesta generativa con Workers AI.
 *
 * Body: { question: string }
 * Respuesta: { answer: string, results: [...], model: string }
 */

import { RagIndex, deserializeIndex } from '../_rag/index.js';

// Modelo Workers AI válido para text generation
const AI_MODEL = '@cf/cloudflare/clairic-flash';

export const onRequestPost = async ({ env, request }) => {
  try {
    const { question } = await request.json();
    if (!question || typeof question !== 'string') {
      return new Response(JSON.stringify({ error: 'Falta "question"' }), {
        status: 400,
        headers: { 'content-type': 'application/json' }
      });
    }

    // 1. Cargar índice desde KV (si existe)
    const index = await loadIndex(env.LINKEDIN_KV);

    // 2. Buscar resultados relevantes
    const results = index.search(question, 8);

    // 3. Construir contexto para la IA
    const context = results.length
      ? results.map(r => `- Fuente: ${r.meta.source} (${r.meta.type})\n  ${r.text}`).join('\n\n')
      : 'No hay contexto disponible en los datos cargados.';

    // 4. Llamar a Workers AI para respuesta generativa
    const prompt = `Eres un asistente experto que responde preguntas sobre los datos de LinkedIn de un usuario.
Tienes acceso a los siguientes documentos extraídos de sus archivos CSV de LinkedIn (Profile, Positions, Education, Skills, Certifications, Invitations, etc.).

Responde en español, de forma clara y concisa. Usa SOLO la información del contexto proporcionado.
Si la respuesta no está en el contexto, di que no lo sabes.
Cita las fuentes (nombre del archivo CSV) cuando sea relevante.
No inventes datos.

Contexto de LinkedIn:
${context}

Pregunta: ${question}`;

    let answer = '';
    let usedModel = null;
    let lastAiError = null;

    // Intento con el modelo principal
    try {
      const aiResponse = await env.AI.run(AI_MODEL, { prompt });
      answer = aiResponse.response || String(aiResponse || '');
      usedModel = AI_MODEL;
    } catch (aiErr) {
      lastAiError = aiErr;
    }

    if (!usedModel) {
      // Workers AI no disponible: devolver resultados locales para que el frontend haga fallback
      return new Response(JSON.stringify({
        answer: null,
        fallback: true,
        error: lastAiError?.message || 'Workers AI no disponible',
        results: results.map(r => ({
          score: r.score,
          name: r.meta.name,
          type: r.meta.type,
          source: r.meta.source,
          text: r.text,
          meta: r.meta
        })),
        message: 'Workers AI no disponible. Se devuelven resultados de búsqueda local.'
      }), {
        headers: { 'content-type': 'application/json' }
      });
    }

    return new Response(JSON.stringify({
      answer,
      model: usedModel,
      results: results.map(r => ({
        score: r.score,
        name: r.meta.name,
        type: r.meta.type,
        source: r.meta.source,
        text: r.text,
        meta: r.meta
      })),
      context
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
    // índice corrupto, ignorar
  }
  return index;
}
