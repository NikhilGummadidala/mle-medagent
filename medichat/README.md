# MediChat frontend

Single-file web client (`index.html`, vanilla HTML/CSS/JS, no build step) for the
MedAgent API. It is served by the backend at `/`; see the repository README for
how to run the app.

- Safety screening before sign-in and again when a message describes an emergency
- Sign-in; chats are stored in the browser per user account
- Health information form shown once after sign-up and editable from the Profile panel; it is saved with the account on the server
- Image attachments (file or camera), each marked as a skin photo or a mammogram
- Replies rendered from escaped text with a small markdown subset

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
