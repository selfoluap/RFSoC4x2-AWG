# Systemd Service Management

The system uses systemd to manage the persistent operation of the software stack, ensuring automatic startup and restart on failure.

## Overview

| Service        | Purpose          | Port |
| -------------- | ---------------- | ---- |
| `awg-backend`  | FastAPI REST API | 8000 |
| `awg-frontend` | Streamlit Web UI | 8501 |

## Installation

Service files are created during installation. If something is not working right, you can check the logs.

YOu can also manage the service via the following commands.

```bash
sudo systemctl start awg-backend awg-frontend
```

```bash
sudo systemctl stop awg-frontend awg-backend
```

```bash
sudo systemctl restart awg-backend
sudo systemctl restart awg-frontend
```

```bash
sudo systemctl status awg-backend
sudo systemctl status awg-frontend
```
