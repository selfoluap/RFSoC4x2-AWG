"""Generated protobuf / gRPC modules for rfsoc_awg.proto.

Uses the stubs published with h2pcontrol when they are installed, otherwise the ones generated
locally by scripts/gen_protos.sh into device_server/gen/.
"""

import sys
from pathlib import Path

try:
    from h2pcontrol.rfsocawg.v1 import rfsoc_awg_pb2 as pb
    from h2pcontrol.rfsocawg.v1 import rfsoc_awg_pb2_grpc as pb_grpc
except ImportError:
    _GEN = Path(__file__).resolve().parent / "gen"
    if not (_GEN / "h2pcontrol" / "rfsocawg" / "v1" / "rfsoc_awg_pb2.py").exists():
        raise ImportError(
            "rfsoc_awg stubs not found: run scripts/gen_protos.sh, or install the stubs "
            "published with h2pcontrol"
        ) from None
    sys.path.insert(0, str(_GEN))
    from h2pcontrol.rfsocawg.v1 import rfsoc_awg_pb2 as pb
    from h2pcontrol.rfsocawg.v1 import rfsoc_awg_pb2_grpc as pb_grpc

__all__ = ["pb", "pb_grpc"]
