"""Plain gRPC device server: the same service, without h2pcontrol.

NOT YET RUN (gRPC is not available where this was written).

    python -m device_server                      # gRPC on :50051, web backend on 127.0.0.1:8001
    python -m device_server --grpc-port 50051 --no-web

Use this to try the service with grpcui or a plain gRPC client, and for control systems other
than h2pcontrol.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import grpc

from device_server.rpcs import RfsocAwgRpcs
from device_server.stubs import pb_grpc
from device_server.web import start_web

MAX_MESSAGE_BYTES = 16 * 1024 * 1024      # LoadSequence carries up to 512 KiB; leave headroom


class StandaloneService(RfsocAwgRpcs, pb_grpc.RfsocAwgServiceServicer):
    pass


async def serve(host: str, grpc_port: int, web: bool) -> None:
    server = grpc.aio.server(options=[
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ])
    pb_grpc.add_RfsocAwgServiceServicer_to_server(StandaloneService(), server)
    server.add_insecure_port(f"{host}:{grpc_port}")
    await server.start()
    logging.info("RFSoC AWG device server listening on %s:%d", host, grpc_port)
    web_task = start_web() if web else None
    try:
        await server.wait_for_termination()
    finally:
        if web_task is not None:
            web_task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description="RFSoC AWG gRPC device server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--grpc-port", type=int, default=50051)
    parser.add_argument("--no-web", action="store_true", help="do not start the web backend")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(args.host, args.grpc_port, not args.no_web))


if __name__ == "__main__":
    main()
