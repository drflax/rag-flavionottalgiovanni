/**
 * Cloudflare Pages Function — POST /api/query
 * RAG para Flavio Nottalgiovanni Sambucetti.
 * Recibe { question, context } y genera respuesta con Workers AI.
 *
 * Body: { question: string, context?: string }
 * Respuesta: { answer: string, model: string }
 */

const AI_MODELS = [
  '@cf/meta/llama-3.2-1b-instruct',
  '@cf/meta/llama-3.2-3b-instruct',
  '@cf/meta/llama-3.1-8b-instruct-fp8',
  '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
  '@cf/meta/llama-4-scout-17b-16e-instruct',
  '@cf/meta/llama-2-7b-chat-hf-lora'
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

    const systemPrompt = `Eres un asistente experto que responde preguntas SOLO sobre ${PERSONA}, basándote en su archivo Markdown (flavio.md).

INSTRUCCIONES IMPORTANTES:
- Responde ÚNICAMENTE sobre ${PERSONA}. No inventes datos sobre otras personas.
- Responde en español, de forma clara y concisa.
- Usa SOLO la información del contexto proporcionado.
- Si la respuesta no está en el contexto, di que no lo sabes.
- No inventes datos, cargos, empresas, fechas ni habilidades.

Contexto de ${PERSONA}:
${context}`;

    const userPrompt = `Pregunta sobre ${PERSONA}: ${question}`;

    let answer = '';
    let usedModel = null;
    let lastAiError = null;

    for (const model of AI_MODELS) {
      // Intento 1: formato prompt
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

      // Intento 2: formato messages
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
      return new Response(JSON.stringify({
        answer: null,
        fallback: true,
        error: lastAiError?.message || 'Todos los modelos de Workers AI fallaron',
        message: 'Workers AI no disponible.'
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
