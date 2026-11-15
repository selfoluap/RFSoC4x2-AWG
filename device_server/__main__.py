"""python -m device_server: gRPC device server plus the web backend in one process.

If the gRPC packages or the generated stubs are missing, only the web backend is started, so
the board's web page keeps working on a system that has not been set up for gRPC yet.
"""

import logging
import os

try:
    from device_server.standalone import main
except ImportError as error:
    logging.basicConfig(level=logging.INFO)
    logging.warning("gRPC device server not started (%s); running the web backend only", error)
    import uvicorn

    uvicorn.run(
        "backend.backend:app",
        host=os.environ.get("RFSOC_BACKEND_HOST", "127.0.0.1"),
        port=int(os.environ.get("RFSOC_BACKEND_PORT", "8001")),
    )
else:
    main()
