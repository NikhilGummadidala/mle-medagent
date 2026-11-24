# MediChat — AI Health Assistant

A clean, minimalist medical AI chatbot web interface built with vanilla HTML/CSS/JavaScript.

## Preview

Open `index.html` in your browser directly, or serve it with any static server:

```bash
python -m http.server 3456 -d C:\Users\angel\Desktop\medichat
# then visit http://localhost:3456
```

## Features

- **Chat interface** — send messages, receive AI responses with typing indicator
- **Conversation sidebar** — create, switch, and delete conversations
- **Profile panel** — editable user profile (name, email, specialty, institution)
- **Settings panel** — model selection, temperature, stream toggle, auto-save, disclaimer toggle
- **Local persistence** — conversations and settings saved to localStorage
- **Mobile responsive** — sidebar collapses on small screens

## File Structure

```
medichat/
└── index.html    # Single-file application (HTML + CSS + JS)
```

## Connect Your AI Backend

Open `index.html`, find the `_callAI` function (around line 430), and replace it with your API:

```javascript
function _callAI(userMsg, cb) {
  // Replace this with your actual API call:
  fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: userMsg })
  })
  .then(res => res.json())
  .then(data => cb(data.reply))
  .catch(err => cb('Sorry, an error occurred.'));
}
```

## Tech Stack

- HTML5 + CSS3 (custom properties, flexbox, animations)
- Vanilla JavaScript (ES5-compatible, no build tools)
- Zero dependencies

## License

MIT


## Sent image previews

Images attached to a user message now remain visible as thumbnails with their
original filenames inside that message. The original File is uploaded through
the existing API; only the display thumbnail is resized (maximum 240 pixels,
original aspect ratio). Thumbnails are stored with the existing browser-local
conversation history and survive a page refresh. Older text-only conversations
remain compatible. Non-image files or images a browser cannot decode retain a
filename attachment card. No API/static-file exposure is needed for this preview.

Local browser component checks (with the app running at localhost:8000):

```sh
pip install -r requirements-test.txt
python -m playwright install chromium
python tests/preview_browser_check.py
```

The component checks use synthetic non-medical fixtures, test multiple attachments,
aspect ratio, older text messages, filename escaping and decode fallback. The
separate end-to-end mammogram evaluation uses actual assigned MIAS files and
real local CNN/Ollama responses; generated fixtures are not used for that evaluation.
