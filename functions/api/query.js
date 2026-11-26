/**
 * Cloudflare Pages Function — POST /api/query
 * RAG para Flavio Nottalgiovanni Sambucetti.
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

    const systemPrompt = `Eres un asistente que responde preguntas SOLO sobre ${PERSONA}.

Tienes este contexto extraído de su archivo flavio.md:
${context}

REGLAS OBLIGATORIAS:
1. Responde SOLO sobre ${PERSONA}.
2. Usa EXCLUSIVAMENTE la información del contexto de arriba.
3. Si el contexto menciona empresas (REVOLVE, Gabpersam, Muki, Digitalgrow, etc.), debes mencionarlas directamente.
4. Si el contexto tiene la respuesta, NO digas "no tengo información".
5. Responde en español, claro y conciso.
6. No inventes datos.

Pregunta: ${question}`;

    let answer = '';
    let usedModel = null;
    let lastAiError = null;

    for (const model of AI_MODELS) {
      try {
        const aiResponse = await env.AI.run(model, {
          prompt: systemPrompt,
          max_tokens: 600,
          temperature: 0.2
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
