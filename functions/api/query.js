/**
 * Cloudflare Pages Function — POST /api/query
 * RAG para Flavio Nottalgiovanni Sambucetti.
 * Recibe { question, context } y genera respuesta con Workers AI.
 */

const AI_MODELS = [
  '@cf/meta/llama-3.2-1b-instruct',   // primero: mejor siguiendo instrucciones
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

REGLAS OBLIGATORIAS:
1. Usa ÚNICAMENTE la información del contexto proporcionado a continuación.
2. Si el contexto menciona empresas (REVOLVE, Gabpersam, Muki, Digitalgrow, etc.), debes mencionarlas directamente.
3. No digas "no tengo información" si el contexto contiene la respuesta.
4. Responde en español, claro y conciso.
5. No inventes datos.

CONTEXTO DE ${PERSONA} (extraído de su perfil LinkedIn):
${context}`;

    const userPrompt = `Pregunta: ${question}`;

    let answer = '';
    let usedModel = null;
    let lastAiError = null;

    for (const model of AI_MODELS) {
      // Intento 1: formato prompt
      try {
        const aiResponse = await env.AI.run(model, {
          prompt: `${systemPrompt}\n\n${userPrompt}`,
          max_tokens: 600,
          temperature: 0.3
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
          ],
          max_tokens: 600,
          temperature: 0.3
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
