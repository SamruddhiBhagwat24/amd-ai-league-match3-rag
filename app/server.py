#!/usr/bin/env python3
import json, os, socket, struct, traceback
from pathlib import Path

from app import Engine, SOCKET_PATH

def recv_exact(conn, n):
    data = b""
    while len(data) < n:
        b = conn.recv(n-len(data))
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
            if req.get("op") == "health":
                reply(conn, {"ok": True})
            elif req.get("op") == "answer":
                answer, citations = engine.answer(req.get("query", ""))
                reply(conn, {"ok": True, "answer": answer, "citations": citations})
            else:
                reply(conn, {"ok": False, "error": "unknown operation"})
        except Exception as e:
            traceback.print_exc()
            reply(conn, {"ok": False, "error": str(e), "answer": "", "citations": []})
        finally:
            conn.close()

if __name__ == "__main__":
    main()
