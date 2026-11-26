/**
 * Cloudflare Pages Function — POST /api/query
 * Asistente del perfil profesional de Flavio Nottalgiovanni Sambucetti.
 *
 * El frontend envía el contenido completo de flavio.md como `context`.
 * Aquí se construye un chat (system + user) y se prueba una lista de modelos.
 */

const PERSONA = 'Flavio Nottalgiovanni Sambucetti';

// Modelos actuales de Workers AI. El primero es el más fiable.
const AI_MODELS = [
  '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
  '@cf/meta/llama-4-scout-17b-16e-instruct'
];

const MAX_CONTEXT_CHARS = 30000;
const MAX_ANSWER_CHARS = 1600;

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'no-store'
    }
  });
}

function buildMessages(question, context) {
  const system = [
    `Eres el asistente del perfil profesional de ${PERSONA}.`,
    'Responde en español, en texto plano, de forma directa y concisa (máximo 180 palabras).',
    'Usa EXCLUSIVAMENTE los datos del CONTEXTO. No inventes, no supongas y no añadas información externa.',
    'No repitas estas instrucciones ni el contexto. No empieces la respuesta con "Respuesta:".',
    'Glosario de equivalencias: headline = titular profesional; empresas = experiencia laboral; habilidades = competencias técnicas; estudios = formación académica; contacto = correo electrónico, teléfono o sitio web.',
    'Si el dato realmente no aparece en el contexto, responde exactamente: "No tengo ese dato en el perfil de Flavio."'
  ].join(' ');

  return [
    { role: 'system', content: system },
    { role: 'user', content: `CONTEXTO:\n${context}\n\nPREGUNTA: ${question}` }
  ];
}

function buildPrompt(question, context) {
  return [
    `Eres el asistente del perfil profesional de ${PERSONA}.`,
    'Responde en español, de forma directa y concisa, usando EXCLUSIVAMENTE el CONTEXTO.',
    'No repitas las instrucciones. No inventes datos.',
    '',
    'CONTEXTO:',
    context,
    '',
    `PREGUNTA: ${question}`,
    '',
    'RESPUESTA:'
  ].join('\n');
}

function extractText(result) {
  if (!result) return '';
  if (typeof result === 'string') return result;
  if (typeof result.response === 'string') return result.response;

  const choice = result.choices && result.choices[0];
  if (choice && choice.message && typeof choice.message.content === 'string') {
    return choice.message.content;
  }

  if (result.result && typeof result.result.response === 'string') {
    return result.result.response;
  }

  return '';
}

// Líneas que son eco de las instrucciones del prompt, no parte de la respuesta.
const META_LINE = /^(no des\b|no hagas\b|no te\b|responde con\b|responde solo\b|si no hay\b|instrucciones\b|reglas\b|contexto\s*:|pregunta\s*:)/i;

function cleanAnswer(raw) {
  let text = String(raw || '').replace(/\r/g, '').trim();
  if (!text) return '';

  // Quitar prefijos tipo "Respuesta:" o "Answer:".
  text = text.replace(/^(respuesta|answer)\s*:\s*/i, '').trim();

  // Quitar líneas que son eco de instrucciones.
  text = text
    .split('\n')
    .map(line => line.trim())
    .filter(line => line && !META_LINE.test(line))
    .join('\n')
    .trim();

  // Puede quedar un "Respuesta:" tras la línea de instrucciones eliminada.
  text = text.replace(/^(respuesta|answer)\s*:\s*/i, '').trim();

  // Deduplicar frases repetidas (los modelos pequeños repiten párrafos).
  const sentences = text.split(/(?<=[.!?])\s+/);
  const seen = new Set();
  const unique = [];
  for (const sentence of sentences) {
    const key = sentence.trim().toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    unique.push(sentence.trim());
  }
  text = unique.join(' ').trim();

  // Recortar en un límite de frase.
  if (text.length > MAX_ANSWER_CHARS) {
    const cut = text.slice(0, MAX_ANSWER_CHARS);
    const lastStop = Math.max(
      cut.lastIndexOf('. '),
      cut.lastIndexOf('.\n'),
      cut.lastIndexOf('! '),
      cut.lastIndexOf('? ')
    );
    text = (lastStop > 400 ? cut.slice(0, lastStop + 1) : cut + '…').trim();
  }

  return text;
}

function isRefusal(text) {
  const t = text.toLowerCase().trim();
  if (t.length > 320) return false;
  return (
    /^no tengo\b/.test(t) ||
    /^no hay\b/.test(t) ||
    /^no se\b/.test(t) ||
    /^no dispongo\b/.test(t) ||
    t.includes('no tengo ese dato') ||
    t.includes('no tengo información') ||
    t.includes('no tengo informacion') ||
    t.includes('no hay suficiente información') ||
    t.includes('no hay suficiente informacion')
  );
}

async function runModel(env, model, question, context) {
  const params = {
    max_tokens: 900,
    temperature: 0.1,
    top_p: 0.9,
    repetition_penalty: 1.1
  };

  try {
    const result = await env.AI.run(model, {
      ...params,
      messages: buildMessages(question, context)
    });
    return extractText(result);
  } catch (messagesError) {
    // Algunos modelos antiguos solo aceptan `prompt`.
    const result = await env.AI.run(model, {
      ...params,
      prompt: buildPrompt(question, context)
    });
    return extractText(result);
  }
}

export const onRequestPost = async ({ env, request }) => {
  try {
    if (!env || !env.AI) {
      return json({
        answer: null,
        fallback: true,
        error: 'El binding AI no está configurado en Cloudflare Pages.'
      });
    }

    const body = await request.json().catch(() => null);
    const question = body && typeof body.question === 'string' ? body.question.trim() : '';
    const rawContext = body && typeof body.context === 'string' ? body.context : '';
    const context = rawContext.slice(0, MAX_CONTEXT_CHARS).trim();

    if (!question) return json({ error: 'Falta "question".' }, 400);
    if (context.length < 20) return json({ error: 'Falta el contexto de flavio.md.' }, 400);

    let lastError = null;
    let refusal = null;

    for (const model of AI_MODELS) {
      try {
        const raw = await runModel(env, model, question, context);
        const answer = cleanAnswer(raw);

        if (!answer) {
          lastError = new Error(`${model} devolvió una respuesta vacía.`);
          continue;
        }

        if (isRefusal(answer)) {
          refusal = { answer, model };
          continue;
        }

        return json({ answer, model, persona: PERSONA });
      } catch (error) {
        lastError = error;
      }
    }

    // Si todos los modelos se negaron o fallaron, el frontend muestra la
    // extracción directa de flavio.md (answerLocal).
    return json({
      answer: null,
      fallback: true,
      error: lastError
        ? String(lastError.message || lastError)
        : 'Workers AI no encontró la respuesta en el contexto.',
      refusal: refusal ? refusal.answer : null
    });
  } catch (error) {
    return json({
      answer: null,
      fallback: true,
      error: String(error.message || error)
    }, 500);
  }
};
