#!/usr/bin/env python3
"""
AMD AI League Match 3 — Background Daemon Server
Maintains pre-loaded model weights and document index in memory.
Exposes a low-latency UNIX domain socket for sub-second query responses.
"""
import json, os, socket, struct, sys, traceback
from pathlib import Path

# Ensure directory of server.py is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import Engine, SOCKET_PATH

def recv_exact(conn, n):
    data = b""
    while len(data) < n:
        b = conn.recv(n - len(data))
        if not b:
            return None
        data += b
    return data

def reply(conn, obj):
    raw = json.dumps(obj, ensure_ascii=False).encode()
    conn.sendall(struct.pack("!I", len(raw)) + raw)

def main():
    SOCKET_PATH.unlink(missing_ok=True)
    engine = Engine()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(SOCKET_PATH))
    os.chmod(SOCKET_PATH, 0o600)
    s.listen(16)
    while True:
        conn, _ = s.accept()
        try:
            header = recv_exact(conn, 4)
            if not header:
                continue
            n = struct.unpack("!I", header)[0]
            raw = recv_exact(conn, n)
            req = json.loads(raw.decode())
            op = req.get("op")
            if op == "health":
                reply(conn, {"ok": True})
            elif op == "reload":
                engine.reload_index()
                reply(conn, {"ok": True})
            elif op == "answer":
                answer, citations, confidence = engine.answer(req.get("query", ""))
                reply(conn, {
                    "ok": True,
                    "answer": answer,
                    "citations": citations,
                    "confidence": confidence
                })
            else:
                reply(conn, {"ok": False, "error": f"unknown operation '{op}'"})
        except Exception as e:
            traceback.print_exc()
            reply(conn, {"ok": False, "error": str(e), "answer": "", "citations": [], "confidence": 0.0})
        finally:
            conn.close()

if __name__ == "__main__":
    main()
