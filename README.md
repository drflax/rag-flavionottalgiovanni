# RAG — Flavio Nottalgiovanni Sambucetti

Asistente web que responde preguntas sobre el perfil profesional de **Flavio Nottalgiovanni Sambucetti** usando un único archivo de conocimiento: [`flavio.md`](flavio.md).

- **Frontend:** HTML + CSS + JavaScript vanilla, desplegado en Cloudflare Pages.
- **Recuperación:** índice TF-IDF en el navegador (para el modo local de respaldo).
- **Generación:** Cloudflare Workers AI a través de `POST /api/query`.
- **Fuente única:** `flavio.md` (~10 KB). El frontend envía el archivo completo como contexto, no solo fragmentos recuperados.

## Estructura

```
.
├── index.html                 # Interfaz del chat
├── style.css                  # Estilos
├── app.js                     # Carga flavio.md, índice TF-IDF, extracción local y llamada a /api/query
├── flavio.md                  # Base de conocimiento (única fuente)
├── functions/
│   └── api/
│       └── query.js           # Pages Function: POST /api/query → Workers AI
├── wrangler.toml              # Config de Cloudflare Pages + binding [ai]
├── _headers                   # Cabeceras de flavio.md
├── package.json
└── LinkedInDataExport/        # LEGADO: CSV de LinkedIn, ya no se usa
```

## Cómo funciona

1. `app.js` descarga `flavio.md` y construye un índice TF-IDF en memoria.
2. Al preguntar, envía `{ question, context }` a `POST /api/query`, donde `context` es **todo** `flavio.md`.
3. `functions/api/query.js` construye un chat (`system` + `user`) y prueba modelos actuales de Workers AI:
   - `@cf/meta/llama-3.3-70b-instruct-fp8-fast` (principal)
   - `@cf/meta/llama-4-scout-17b-16e-instruct` (respaldo)
4. Si Workers AI falla o se niega a responder, el frontend usa un **extractor directo** de `flavio.md` que responde a las preguntas habituales: empresas, habilidades, formación, titular profesional, contacto, ubicación, gestión deportiva y docencia.
5. Si el dato no existe en `flavio.md`, la app lo dice explícitamente; no inventa.

## Modelos de Workers AI

Los modelos se configuran en `functions/api/query.js` (`AI_MODELS`). Cloudflare deprecia modelos con frecuencia; si aparece un error `5028`, cambia el ID por uno vigente del catálogo:

https://developers.cloudflare.com/workers-ai/models/

## Despliegue en Cloudflare Pages

1. Conecta el repositorio a Cloudflare Pages.
2. Configuración de build:
   - Framework preset: `None`
   - Build command: vacío
   - Build output directory: `.`
3. Asegúrate de que `wrangler.toml` incluya:

```toml
pages_build_output_dir = "."

[ai]
binding = "AI"
```

4. Cada push a `main` despliega automáticamente.

## Desarrollo local

Requiere Node.js 22+ y Wrangler:

```bash
npm install
npx wrangler pages dev .
```

Abre `http://localhost:8788`.

## Privacidad

- La app solo usa `flavio.md`.
- `LinkedInDataExport/` contiene datos personales de la exportación de LinkedIn y **ya no se usa**. Está en `.gitignore`, pero algunos archivos siguen en el historial de Git. Si el repositorio es público, considera eliminarlos del seguimiento y reescribir el historial.
- `flavio.md` incluye email, teléfono y ubicación porque el perfil los publica; revísalo antes de hacer público el sitio.
