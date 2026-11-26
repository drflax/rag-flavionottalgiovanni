/**
 * Cloudflare Pages Function — POST /api/query
 * RAG para Flavio Nottalgiovanni Sambucetti.
 */

const AI_MODELS = [
  '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
  '@cf/meta/llama-4-scout-17b-16e-instruct'
];

const PERSONA = 'Flavio Nottalgiovanni Sambucetti';

export const onRequestPost = async ({ env, request }) => {
  try {
    const { question, context = '' } = await request.json();
    if (!question || typeof question !== 'string') {
      return new Response(JSON.stringify({ error: 'Falta "question"' }), {
        status: 400,
        headers: { 'content-type': 'application/json' }
      });
    }

    const prompt = `Responde sobre ${PERSONA} usando SOLO este contexto:

${context}

Pregunta: ${question}

Reglas: Responde en español. Si el contexto tiene la respuesta, dila directamente. No digas "no tengo información" si el contexto la contiene. No inventes datos.`;

    let answer = '';
    let usedModel = null;
    let lastAiError = null;

    for (const model of AI_MODELS) {
      try {
        const aiResponse = await env.AI.run(model, {
          prompt,
          max_tokens: 600,
          temperature: 0.1
        });
        answer = aiResponse.response || String(aiResponse || '');
        usedModel = model;
        break;
      } catch (aiErr) {
        lastAiError = aiErr;
      }
    }

    if (!usedModel) {
      return new Response(JSON.stringify({
        answer: null,
        fallback: true,
        error: lastAiError?.message || 'Todos los modelos fallaron'
      }), {
        headers: { 'content-type': 'application/json' }
      });
    }

    return new Response(JSON.stringify({
      answer,
      model: usedModel,
      persona: PERSONA
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
