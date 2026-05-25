## Backend Architecture Boundaries

- **API layer (`backend/backend.py`)**: owns FastAPI request/response models, dependency wiring, and endpoint orchestration.
- **Reusable library layer (`backend/services/`)**: owns backend waveform orchestration and the hardware service wrapper around `OverlayController`.
- **Integration guidance (frontend/notebooks/tests)**: import reusable functionality from `backend.services` instead of calling route-handler helper functions in `backend/backend.py`.

This split keeps HTTP concerns isolated from reusable control and signal-processing logic.
