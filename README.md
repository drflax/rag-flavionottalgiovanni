# RAG LinkedIn CSV

Proyecto ligero de **RAG (Retrieval-Augmented Generation)** que se nutre de archivos **CSV exportados de LinkedIn** (contactos, experiencia, educación, certificaciones, habilidades, invitaciones…). Está pensado para desplegar como frontend en **Cloudflare Pages**, con opción de respuestas generativas vía **Workers AI**.

## Estructura

```
.
├── index.html              # Frontend (HTML + CSS + JS vanilla, responsive, español)
├── style.css               # Estilos
├── app.js                  # Motor TF-IDF + coseno, parseo de CSV, UI y llamada a Workers AI
├── wrangler.toml           # Config de Cloudflare Pages / Functions
├── package.json
├── README.md
├── .gitignore
├── LinkedInDataExport/     # <-- Tus CSV de LinkedIn van aquí (no se suben al repo)
│   ├── Profile.csv
│   ├── Positions.csv
│   ├── Education.csv
│   ├── Skills.csv
│   ├── Certifications.csv
│   ├── Invitations.csv
│   ├── Company Follows.csv
│   └── … (todos los CSV de tu exportación de LinkedIn)
└── functions/              # API server-side (opcional, para Cloudflare Pages Functions)
    ├── _rag/
    │   └── index.js        # Código compartido: motor TF-IDF + parseadores de CSV
    └── api/
        ├── query.js        # POST /api/query  → búsqueda + Workers AI
        └── ingest.js       # POST /api/ingest → indexar CSV en KV
```

## Cómo funciona (modo local, sin backend)

1. Abre `index.html` en el navegador (o despliega en Cloudflare Pages como sitio estático).
2. Sube uno o más CSV de tu carpeta `LinkedInDataExport/`.
3. `app.js` parsea cada fila (perfil, posiciones, educación, habilidades, certificaciones, invitaciones, notas, emails, teléfonos, empresas seguidas…) y construye un índice **TF-IDF + coseno** en memoria.
4. Escribe una pregunta en español, ej.:
   - *"¿en qué empresas ha trabajado Flavio?"*
   - *"¿qué habilidades tiene?"*
   - *"¿dónde estudió?"*
   - *"contactos de marketing"*
5. El motor devuelve los resultados más relevantes con puntuación y enlace al perfil.

Todo ocurre **en el navegador**; los datos no salen de tu máquina.

## Integración con Workers AI (respuestas generativas)

Si desplegas en Cloudflare Pages con Functions, puedes usar **Workers AI** para generar respuestas en lenguaje natural basadas en el contexto recuperado.

### Pasos

1. **Despliega las funciones** en Cloudflare Pages (conecta tu repo de GitHub o usa `wrangler pages deploy`).
2. **Crea un KV Namespace** (opcional, para persistencia del índice):
   - Dashboard de Cloudflare → Workers & Pages → KV → Crear namespace
   - En `wrangler.toml` descomenta la sección `[[kv_namespaces]]` y pon el `id`:
     ```toml
     [[kv_namespaces]]
     binding = "LINKEDIN_KV"
     id = "tu-kv-id-aqui"
     ```
3. **Modelo**: por defecto se usa `@cf/meta/llama-3.1-8b-instruct`. Puedes cambiarlo en `functions/api/query.js`.
4. Los endpoints disponibles son:
   - `POST /api/ingest` — sube un CSV (campo `file`) y lo indexa en KV.
   - `POST /api/query` — envía `{ "question": "..." }` y obtiene una respuesta generativa con Workers AI + los resultados de búsqueda como fuentes.

### Flujo con Workers AI

```
Usuario pregunta → /api/query → búsqueda TF-IDF en índice (KV o local)
                                    ↓
                              Contexto relevante + pregunta → Workers AI (Llama 3.1 8B)
                                    ↓
                          Respuesta generativa en español + citas de fuentes
```

Si Workers AI no está disponible, el frontend hace **fallback automático** al modo local (resultados de búsqueda sin respuesta generativa).

## Desplegar en Cloudflare Pages

### Opción A: Solo frontend estático (rápido, sin backend)

1. Sube este proyecto a GitHub (sin la carpeta `LinkedInDataExport/`, contiene tus datos privados).
2. Conecta el repo a **Cloudflare Pages**.
3. Build settings:
   - Framework preset: `None`
   - Build command: vacío
   - Build output directory: `.`
4. Despliega. Tu sitio será `https://<proyecto>.pages.dev`.
5. En el navegador, sube tus CSV desde `LinkedInDataExport/`.

### Opción B: Con backend (Workers AI + KV persistente)

1. Sube el proyecto a GitHub (sin `LinkedInDataExport/`).
2. Conecta a Cloudflare Pages.
3. Crea un KV Namespace y configúralo en `wrangler.toml` (`LINKEDIN_KV`).
4. Despliega con Wrangler:
   ```bash
   cd /Users/drf/rag-flavionottalgiovanni
   npm install
   npx wrangler pages deploy .
   ```
5. Usa `POST /api/ingest` para indexar CSV y `POST /api/query` para preguntas con IA.

### Desarrollo local

```bash
cd /Users/drf/rag-flavionottalgiovanni
npm install
npx wrangler pages dev .
```
Luego abre `http://localhost:8788`.

## CSVs de LinkedIn soportados

| Archivo | Qué se indexa |
|---------|--------------|
| `Profile.csv` | Perfil, headline, resumen, industria, ubicación, webs |
| `Positions.csv` | Empresas, cargos, ubicaciones, fechas, descripciones |
| `Education.csv` | Instituciones, títulos, fechas |
| `Skills.csv` | Habilidades |
| `Certifications.csv` | Certificaciones, autoridad, URLs |
| `Invitations.csv` | Contactos (de/para, fecha, mensaje) |
| `Company Follows.csv` | Empresas seguidas |
| `Notes.csv` | Notas sobre contactos |
| `Email Addresses.csv` | Emails |
| `PhoneNumbers.csv` | Teléfonos |
| `Organizations.csv` | Organizaciones |
| `Rich_Media.csv` | Enlaces multimedia |
| `Whatsapp Phone Numbers.csv` | WhatsApp |
| `Registration.csv` | Datos de registro |
| `SavedJobAlerts.csv` | Alertas de empleo |

## Limitaciones

- **Modo local**: el índice vive en memoria y se reinicia al recargar la página. Para persistencia real, usa `functions/api/` con KV.
- El motor base es TF-IDF + coseno (búsqueda semántica ligera, sin claves de API ni LLM externa).
- Para respuestas generativas, necesitas desplegar en Cloudflare Pages con Functions y Workers AI habilitado.

## Requerimientos

- Navegador moderno (FileReader, Fetch).
- Node.js + Wrangler solo si usas el backend de Cloudflare Functions.
