"""RFSoC AWG device server for h2pcontrol.

TEMPLATE - NOT YET RUN. It follows the pattern of the MccDaqService example: a ``Server``
subclass with ``_healthy`` and async RPC methods, started with
``asyncio.run(Service(ServerConfig.load()).start())``. The SDK import below is my reading of
that example, not of the SDK itself: adjust it to the real h2pcontrol-sdk.

The RPC methods come from rpcs.py and are shared with the plain gRPC server (standalone.py),
which runs without h2pcontrol.

Run on the board:
    python -m device_server.service
"""

from __future__ import annotations

import asyncio

# --- adjust this import to the real SDK ------------------------------------------------------
from h2pcontrol_sdk import Server, ServerConfig
# ---------------------------------------------------------------------------------------------

from device_server.rpcs import RfsocAwgRpcs
from device_server.stubs import pb_grpc
from device_server.web import start_web


class RfsocAwgService(Server, RfsocAwgRpcs, pb_grpc.RfsocAwgServiceServicer):
    def _healthy(self) -> bool:
        return self._core is not None


async def main() -> None:
    web = start_web()          # the board's status page, in this process, sharing the device
    try:
        await RfsocAwgService(ServerConfig.load()).start()
    finally:
        if web is not None:
            web.cancel()


if __name__ == "__main__":
    asyncio.run(main())
