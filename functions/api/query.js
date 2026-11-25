/**
 * Cloudflare Pages Function — POST /api/query
 * Búsqueda en el índice (KV si está disponible) + respuesta generativa con Workers AI.
 *
 * Body: { question: string }
 * Respuesta: { answer: string, results: [...], model: string }
 */

import { RagIndex, deserializeIndex } from '../_rag/index.js';

// Modelos de Workers AI vigentes para texto generativo (catálogo 2026)
// Se prueban en orden hasta que uno funcione.
const AI_MODELS = [
  '@cf/meta/llama-3.2-1b-instruct',
  '@cf/meta/llama-3.2-3b-instruct',
  '@cf/meta/llama-3.1-8b-instruct-fp8',
  '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
  '@cf/meta/llama-4-scout-17b-16e-instruct',
  '@cf/meta/llama-2-7b-chat-hf-lora'
];

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

    // 4. Llamar a Workers AI (intenta varios modelos y formatos)
    const systemPrompt = `Eres un asistente experto que responde preguntas sobre los datos de LinkedIn de un usuario.
Tienes acceso a los siguientes documentos extraídos de sus archivos CSV de LinkedIn (Profile, Positions, Education, Skills, Certifications, Invitations, etc.).

Responde en español, de forma clara y concisa. Usa SOLO la información del contexto proporcionado.
Si la respuesta no está en el contexto, di que no lo sabes.
Cita las fuentes (nombre del archivo CSV) cuando sea relevante.
No inventes datos.`;

    const userPrompt = `Contexto de LinkedIn:\n\n${context}\n\nPregunta: ${question}`;

    let answer = '';
    let usedModel = null;
    let lastAiError = null;

    for (const model of AI_MODELS) {
      // Intento 1: formato prompt (más compatible)
      try {
        const aiResponse = await env.AI.run(model, {
          prompt: `${systemPrompt}\n\n${userPrompt}`,
          max_tokens: 512,
          temperature: 0.7
        });
        answer = aiResponse.response || String(aiResponse || '');
        usedModel = model;
        break;
      } catch (aiErr) {
        lastAiError = aiErr;
      }

      // Intento 2: formato messages (para modelos chat)
      try {
        const aiResponse = await env.AI.run(model, {
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
          ]
        });
        answer = aiResponse.response || String(aiResponse || '');
        usedModel = model;
        break;
      } catch (aiErr2) {
        lastAiError = aiErr2;
        continue;
      }
    }

    if (!usedModel) {
      // Workers AI no disponible: devolver resultados locales para que el frontend haga fallback
      return new Response(JSON.stringify({
        answer: null,
        fallback: true,
        error: lastAiError?.message || 'Todos los modelos de Workers AI fallaron',
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
