# Frontend v2 (Vite + TypeScript + Plotly.js)

A second frontend implementation for RFSoC4x2-AWG, built with:

- Vite
- TypeScript
- React + React Router
- Plotly.js (`plotly.js-dist-min`)

## Run locally

```bash
cd frontend-v2
npm install
npm run dev
```

## Routing / pages

- `/` Dashboard (status/constants/capture quick actions)
- `/serrodyne` Serrodyne waveform page
- `/simple` Simple waveform page
- `/analysis` Error metrics page

## Backend configuration

Default backend URL is `http://localhost:8001`.

Override full URL:

```bash
VITE_RFSOC_BACKEND_URL=http://<host>:8001 npm run dev
```

Override only port:

```bash
VITE_RFSOC_BACKEND_PORT=8001 npm run dev
```
