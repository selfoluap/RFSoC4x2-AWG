# Frontend v2 (Vite + TypeScript + Plotly.js)

A second frontend implementation for RFSoC4x2-AWG, built with:

- Vite
- TypeScript
- React + React Router
- Plotly.js (`plotly.js-dist-min`)

## Run locally

```bash
cd frontend
npm install
npm run dev
```

## Build

```bash
cd frontend
npm install
npm run build
```

The production build is written to `frontend/dist` and is served by nginx on the RFSoC board.

## Routing / pages

- `/` Dashboard (status/constants/capture quick actions)
- `/serrodyne` Serrodyne waveform page
- `/simple` Simple waveform page
- `/analysis` Error metrics page

## Backend configuration

Default backend URL is `/api`, which lets nginx proxy requests to the backend service.

Override full URL:

```bash
VITE_RFSOC_BACKEND_URL=http://<host>:8001 npm run dev
```
