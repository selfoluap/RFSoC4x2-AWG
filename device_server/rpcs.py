"""RPC methods of the RFSoC AWG device service.

NOT YET RUN: gRPC is not available where this was written. Every method is a few lines:
translate the request (convert.py), call the device core (firmware/device.py), translate the
answer. The behaviour lives in those two modules, which are tested without gRPC.
"""

from __future__ import annotations

import asyncio

import grpc

from device_server import convert
from device_server.stubs import pb
from firmware.device import DeviceCore, get_device


async def _call(context, function, *args, **kwargs):
    """Run a device-core call in a thread and map its errors to gRPC status codes."""
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    except (ValueError, TypeError, KeyError) as error:
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
    except RuntimeError as error:          # DeviceError and the players' own state errors
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))


def _output(status: dict) -> "pb.OutputStatus":
    return pb.OutputStatus(**convert.output_status_fields(status))


class RfsocAwgRpcs:
    """The RPC methods of RfsocAwgService, independent of how the server is hosted.

    Mixed into the h2pcontrol server (service.py) and the plain gRPC server (standalone.py).
    """

    _core: DeviceCore | None = None

    def _device(self) -> DeviceCore:
        """The device core; the first use loads the overlay, which takes a few seconds."""
        if self._core is None:
            self._core = get_device()
        return self._core

    async def GetInfo(self, request, context):
        info = await _call(context, lambda: self._device().info())
        return pb.GetInfoResponse(
            dac_sample_rate_hz=info["dac_sample_rate_hz"],
            capabilities=pb.Capabilities(**info["capabilities"]),
            channels=[pb.ChannelInfo(**convert.channel_info_fields(c)) for c in info["channels"]],
            bitfile=info["bitfile"],
            clock_source=info["clock_source"],
        )

    async def GetStatus(self, request, context):
        status = await _call(context, lambda: self._device().status())
        return pb.GetStatusResponse(outputs=[_output(s) for s in status["outputs"]])

    async def LoadLoop(self, request, context):
        def work():
            return self._device().load_loop(
                convert.channel_name(request.channel), convert.samples_from_bytes(request.samples)
            )

        return pb.LoadLoopResponse(output=_output(await _call(context, work)))

    async def LoadSequence(self, request, context):
        def work():
            return self._device().load_sequence(
                convert.channel_name(request.channel),
                convert.blocks_from_bytes(request.samples, request.block_samples),
            )

        return pb.LoadSequenceResponse(output=_output(await _call(context, work)))

    async def LoadWaveform(self, request_iterator, context):
        # The stream is read here, in the event loop, and handed to the loader thread through
        # a queue, so a long waveform never has to sit in memory as a whole.
        stream = convert.WaveformStream()
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        loop = asyncio.get_running_loop()

        def chunks():
            while True:
                item = asyncio.run_coroutine_threadsafe(queue.get(), loop).result()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item

        async def feed():
            try:
                async for message in request_iterator:
                    if message.WhichOneof("part") == "header":
                        header = message.header
                        stream.header(header.channel, header.sample_bits, header.interpolation)
                    else:
                        await queue.put(stream.chunk(message.samples))
                await queue.put(None)
            except Exception as error:
                await queue.put(error)

        first = await request_iterator.__anext__()
        if first.WhichOneof("part") != "header":
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                "LoadWaveform: the first message must be the header")
        try:
            stream.header(first.header.channel, first.header.sample_bits, first.header.interpolation)
        except ValueError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

        feeder = asyncio.ensure_future(feed())
        try:
            status = await _call(
                context,
                lambda: self._device().load_long(
                    stream.channel, chunks(),
                    sample_bits=stream.sample_bits, interpolation=stream.interpolation,
                ),
            )
        finally:
            feeder.cancel()
        return pb.LoadWaveformResponse(output=_output(status))

    async def Arm(self, request, context):
        def work():
            return self._device().arm(
                convert.channel_name(request.channel),
                trigger=convert.trigger_name(request.trigger),
                repeats=request.repeats or 1,
                end=convert.end_name(request.end),
            )

        return pb.ArmResponse(output=_output(await _call(context, work)))

    async def Trigger(self, request, context):
        status = await _call(
            context, lambda: self._device().trigger(convert.channel_name(request.channel))
        )
        return pb.TriggerResponse(output=_output(status))

    async def Stop(self, request, context):
        status = await _call(
            context, lambda: self._device().stop(convert.channel_name(request.channel))
        )
        return pb.StopResponse(output=_output(status))

    async def StopAll(self, request, context):
        status = await _call(context, lambda: self._device().stop_all())
        return pb.StopAllResponse(outputs=[_output(s) for s in status["outputs"]])
