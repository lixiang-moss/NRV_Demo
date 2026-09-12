"""Local, lossless framing for the ROS-to-host E2FAI integration.

NEF1 is independent of the research branch's NRV transport. Event timestamps
are the original /dvs/events integer nanoseconds, never reconstructed here.
"""
import json
import socket
import struct

import numpy as np

PROTOCOL_VERSION = 1
HEADER = struct.Struct("!4sIQ")
MAGIC = b"NEF1"
MAX_METADATA_BYTES = 65536
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
EVENT_DTYPE = np.dtype([
    ("x", "<u2"), ("y", "<u2"), ("timestamp_ns", "<u8"), ("polarity", "u1")
])


class ProtocolError(ValueError):
    pass


def send_packet(connection, kind, metadata, payload=b""):
    if kind not in ("events", "result", "error") or not isinstance(metadata, dict):
        raise ProtocolError("Invalid packet kind or metadata")
    encoded = json.dumps({"kind": kind, "meta": metadata},
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    view = memoryview(payload).cast("B")
    if len(encoded) > MAX_METADATA_BYTES or len(view) > MAX_PAYLOAD_BYTES:
        raise ProtocolError("Packet exceeds protocol size limit")
    connection.sendall(HEADER.pack(MAGIC, len(encoded), len(view)))
    connection.sendall(encoded)
    if view:
        connection.sendall(view)


def _read_exact(connection, size, clean_eof=False):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        try:
            count = connection.recv_into(view[offset:])
        except socket.timeout:
            # A timeout between fragments must not discard a partial frame.
            # The owner interrupts a session by shutting down its socket.
            continue
        if not count:
            if clean_eof and offset == 0:
                return None
            raise EOFError("Connection ended inside a packet")
        offset += count
    return data


def recv_packet(connection):
    header = _read_exact(connection, HEADER.size, clean_eof=True)
    if header is None:
        return None
    magic, metadata_size, payload_size = HEADER.unpack(header)
    if magic != MAGIC or not 0 < metadata_size <= MAX_METADATA_BYTES:
        raise ProtocolError("Invalid protocol header")
    if payload_size > MAX_PAYLOAD_BYTES:
        raise ProtocolError("Payload exceeds protocol size limit")
    try:
        envelope = json.loads(_read_exact(connection, metadata_size).decode("utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ProtocolError("Invalid metadata JSON") from error
    if (not isinstance(envelope, dict)
            or envelope.get("kind") not in ("events", "result", "error")
            or not isinstance(envelope.get("meta"), dict)):
        raise ProtocolError("Invalid packet metadata")
    return envelope["kind"], envelope["meta"], _read_exact(connection, payload_size)
