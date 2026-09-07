import json
from pathlib import Path as _P
_info = json.loads(_P("/kaggle/working/server_info.json").read_text()) \
    if _P("/kaggle/working/server_info.json").exists() else {}
if not _info.get("healthy"):
    print("SKIP: server not healthy -",
          _info.get("reason", "no model served yet"))
    _P("/kaggle/working/prompts_results.json").write_text(
        json.dumps({"skipped": True,
                    "reason": _info.get("reason", "no model served yet")}))
else:
    import json, time
    from pathlib import Path
    import requests

    BASE = "http://localhost:8000"
    MODEL = requests.get(f"{BASE}/v1/models", timeout=60).json()["data"][0]["id"]
    print("serving model id:", MODEL)

    def run_once(prompt_text, max_tokens, timeout=3600):
        t0 = time.time()
        r = requests.post(f"{BASE}/v1/completions",
                          json={"model": MODEL, "prompt": prompt_text,
                                "max_tokens": max_tokens, "temperature": 0.0,
                                "stream": True,
                                "stream_options": {"include_usage": True}},
                          timeout=timeout, stream=True)
        r.raise_for_status()
        t_first, usage, texts = None, {}, []
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            else:
                for ch in obj.get("choices", []):
                    t = ch.get("text", "")
                    if t:
                        texts.append(t)
                        if t_first is None:
                            t_first = time.time()
        t_end = time.time()
        pt = usage.get("prompt_tokens", -1)
        ct = usage.get("completion_tokens", -1)
        ttft = (t_first or t_end) - t0
        return {"prompt_tokens": pt, "completion_tokens": ct,
                "ttft_s": round(ttft, 3), "total_s": round(t_end - t0, 3),
                "prefill_tok_per_s": round(pt / max(ttft, 1e-9), 1),
                "decode_tok_per_s": round(
                    ct / max(t_end - (t_first or t_end), 1e-9), 1),
                "text": "".join(texts)[:3000]}

    print("warmup ...", flush=True)
    print(run_once("Explain superconductivity in two sentences.", 64))

    P1 = "Explain superconductivity in two sentences."
    P2 = ("A farmer has 17 sheep. All but 9 run away. A truck then delivers "
          "twice as many sheep as remain, and a storm scatters a third of the "
          "flock (rounded down). How many sheep are left? Show each step.")
    P3_HEAD = ("You are a senior Python reviewer. The multithreaded task queue "
               "below has exactly one concurrency bug that can lose tasks under "
               "load. Identify the buggy lines, explain the race, and provide "
               "the minimal fix. Here is the code:\n\n")
    P3_MOD_A = '''
    import queue, threading, time

    class TaskQueue:
        """Bounded multi-producer multi-consumer task queue with stats."""

        def __init__(self, maxsize=1000, num_workers=4):
            self._q = queue.Queue(maxsize=maxsize)
            self._num_workers = num_workers
            self._workers = []
            self._shutdown = False
            self._pending = 0
            self._finished = 0
            self._errors = []
            self._lock = threading.Lock()
            self._stats = {"submitted": 0, "completed": 0, "retried": 0}

        def start(self):
            for i in range(self._num_workers):
                t = threading.Thread(target=self._worker, args=(i,),
                                     daemon=True, name=f"worker-{i}")
                t.start()
                self._workers.append(t)

        def submit(self, fn, *args, **kwargs):
            with self._lock:
                self._pending += 1
                self._stats["submitted"] += 1
            self._q.put((fn, args, kwargs))

        def _worker(self, wid):
            while True:
                try:
                    fn, args, kwargs = self._q.get(timeout=0.5)
                except queue.Empty:
                    with self._lock:
                        if self._shutdown and self._pending == 0:
                            return
                    continue
                try:
                    fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - record and continue
                    with self._lock:
                        self._errors.append((wid, repr(exc)))
                finally:
                    self._pending -= 1
                    self._finished += 1
                    with self._lock:
                        self._stats["completed"] += 1
                    self._q.task_done()

        def shutdown(self, timeout=30.0):
            with self._lock:
                self._shutdown = True
            deadline = time.time() + timeout
            for t in self._workers:
                remaining = max(0.0, deadline - time.time())
                t.join(timeout=remaining)
            return self.snapshot()

        def snapshot(self):
            with self._lock:
                return dict(pending=self._pending, finished=self._finished,
                            errors=list(self._errors), stats=dict(self._stats))

        def wait_empty(self, timeout=60.0):
            if not self._q.join():
                pass
            start = time.time()
            while self._pending and time.time() - start < timeout:
                time.sleep(0.05)
            return self._pending == 0
    '''
    P3_MOD_B = '''
    """Retry helpers and backpressure policies used by the queue above."""

    import random
    import time


    def retry_with_backoff(fn, attempts=3, base=0.1, cap=5.0, jitter=True):
        last = None
        for i in range(attempts):
            try:
                return fn()
            except Exception as exc:
                last = exc
                delay = min(cap, base * (2 ** i))
                if jitter:
                    delay = random.uniform(0, delay)
                time.sleep(delay)
        raise last


    class TokenBucket:
        def __init__(self, rate, capacity):
            self.rate = rate
            self.capacity = capacity
            self.tokens = capacity
            self.updated = time.monotonic()

        def allow(self, n=1):
            now = time.monotonic()
            self.tokens = min(self.capacity,
                              self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False


    def chunked(iterable, n):
        buf = []
        for item in iterable:
            buf.append(item)
            if len(buf) >= n:
                yield buf
                buf = []
        if buf:
            yield buf


    def moving_average(values, window):
        if window <= 0:
            raise ValueError("window must be positive")
        out, total = [], 0.0
        for i, v in enumerate(values):
            total += v
            if i >= window:
                total -= values[i - window]
            out.append(total / min(i + 1, window))
        return out
    '''
    P3 = P3_HEAD + P3_MOD_A + P3_MOD_B * 6
    P4 = ("Write a production-quality async Python web crawler with per-domain "
          "rate limiting, robots.txt respect, retries with backoff, and SQLite "
          "persistence. Return only the code with a short usage example.")

    results = {"model_id": MODEL, "prompts": []}
    for label, prompt, mtok in (("short_factual", P1, 128),
                                ("reasoning", P2, 256),
                                ("long_code_review", P3, 512),
                                ("codegen", P4, 512)):
        print(f"--- {label} (prompt chars: {len(prompt)}) ---", flush=True)
        m = run_once(prompt, mtok)
        m["label"] = label
        print({k: (v if k != "text" else v[:400]) for k, v in m.items()},
              flush=True)
        results["prompts"].append(m)

    Path("/kaggle/working/prompts_results.json").write_text(json.dumps(results, indent=1))
    print("PROMPT TESTS COMPLETE")
