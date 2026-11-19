# MedAgent / MediChat

A prototype health assistant for the general public: a chat interface over three
specialist ML models (heart disease likelihood, mammogram screening, skin lesion
classification) plus a local LLM for general health questions and plain-language
explanations.

**This is a prototype, not a medical device. None of the models is clinically
validated.** See `saved_models/README.md` for what each model is, where it came
from and how well it measured.

## Run it

```sh
pip install -r requirements.txt
python scripts/download_models.py        # ~600 MB of open-source weights, checksum-verified
ollama serve                             # in another terminal
ollama pull llama3.2
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. The model assessments work without Ollama; general
answers and explanations need it.

Optional environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEDAGENT_SECRET_KEY` | random key saved in `data/secret.key` | JWT signing key |
| `MEDAGENT_TEXT_MODEL` | `llama3.2` | Ollama model |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `MEDAGENT_CORS_ORIGINS` | none (same-origin only) | comma-separated allowed origins |
| `MEDAGENT_DATA_DIR` | `data/` | user database and signing key |

## How a message is handled

`langgraph_multi_agent.py`:

1. **Extract** measurements from the message. An LLM proposes them; only values
   that are in range and literally present in the patient's words are kept.
   Plain patterns (age, sex, blood pressure, cholesterol, "no chest pain") back
   the LLM up, and chest pain is classified from three yes/no facts by rule.
2. **Plan** once which specialists run. A newly attached image or a stated
   cardiac measurement always goes to its model. Each specialist runs at most once.
3. **Specialists** run the models in `models/`. The LLM only explains a result a
   model produced; it never interprets an image and never substitutes for a
   missing model.
4. **Synthesis** builds the reply from the model outputs. If no specialist
   applies, a general-answer node responds, and it is blocked from writing
   anything that looks like an assessment.

Emergency wording is screened in the browser before sending and again on the
server (`safety.py`), which puts a 911 notice at the top of the reply.

Each account has a health profile (age, sex, conditions, medications, allergies,
usual blood pressure ...) stored with the user in the database (`api/profiles.py`).
It is asked for once after sign-up, edited from the Profile panel, and attached
by the server to every chat turn; the client cannot override it per message.

Measurements and uploaded images persist across the turns of a conversation.
Conversations are held in memory and are lost when the server restarts.

## Layout

| Path | What |
| --- | --- |
| `api/` | FastAPI app: auth, health profiles, routes, graph runner |
| `langgraph_multi_agent.py` | conversation graph |
| `models/` | model inference; `models/schema.py` is the single definition of the tabular inputs |
| `safety.py` | server-side emergency screening |
| `ingestion.py` | upload validation and storage |
| `medichat/index.html` | single-file frontend |
| `scripts/` | model download, training and evaluation |
| `saved_models/` | model files and their documentation |
| `diag-agents/` | original exploration notebooks and the heart dataset (the notebooks are superseded by `scripts/`) |
| `tests/` | pytest suite |

## Security notes

- Every endpoint except register/login needs a bearer token. Conversations and
  uploads are scoped to the signed-in user.
- Uploads are verified as images, capped at 10 MB, stored under random names,
  referenced by opaque ids, and deleted after 24 hours.
- The browser stores chats in `localStorage` per user account.
- Not done yet: login rate limiting, HTTPS (run behind a TLS proxy), persistent
  conversation storage, and password reset.

## Tests

```sh
pip install -r requirements-test.txt
python -m pytest tests
```

The LLM is stubbed in the tests. `tests/test_live_regressions.py` pins faults
that only appeared when running against the real `llama3.2`.
`tests/preview_browser_check.py` is a separate Playwright check that needs the
server running on port 8000.
