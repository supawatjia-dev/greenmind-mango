import asyncio, time, json, random, httpx, statistics

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "mango-awq"
RATE = 4            # request ต่อวินาที
DURATION = 300

SHORT = ["อธิบายเรื่อง KV cache สั้นๆ", "Write a haiku about GPUs",
         "แก้สมการ 3x + 7 = 25 พร้อมแสดงวิธีทำ", "Explain TCP vs UDP briefly"]
LONG_DOC = ("The quick brown fox jumps over the lazy dog. " * 600)  # ~6K tokens
def make_prompt():
    r = random.random()
    if r < 0.7:
        return random.choice(SHORT), 100
    return f"Summarize this document:\n{LONG_DOC}", 6000

results = []

async def one(client, prompt, ptoks):
    t0 = time.perf_counter()
    ttft = last = None
    n = 0
    finish = None
    body = {"model": MODEL, "stream": True, "max_tokens": 8192, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    try:
        async with client.stream("POST", URL, json=body, timeout=900) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                ch = json.loads(line[6:])
                if not ch.get("choices"):
                    continue
                c = ch["choices"][0]
                d = c.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    last = now
                    n += 1
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
    except Exception as e:
        finish = f"error:{type(e).__name__}"
    tpot = (last - t0 - ttft) / max(n - 1, 1) if ttft and n > 1 else None
    slo = 0.5 + 0.25 * ptoks / 1000
    ok = (ttft is not None and ttft <= slo and finish == "stop"
          and (tpot is None or tpot <= 0.05))
    results.append(dict(ttft=ttft, tpot=tpot, finish=finish, ok=ok))

async def main():
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1000)) as client:
        tasks, start = [], time.time()
        while time.time() - start < DURATION:
            p, pt = make_prompt()
            tasks.append(asyncio.create_task(one(client, p, pt)))
            await asyncio.sleep(1 / RATE)
        await asyncio.gather(*tasks)
    ttfts = sorted(r["ttft"] for r in results if r["ttft"])
    tpots = sorted(r["tpot"] for r in results if r["tpot"])
    q = lambda xs: xs[int(len(xs) * 0.95)] if xs else None
    print("requests:", len(results), " passed SLO:", sum(r["ok"] for r in results))
    print("p95 TTFT:", q(ttfts), " p95 TPOT:", q(tpots))
    print("finish:", {k: sum(1 for r in results if r["finish"] == k)
                      for k in set(r["finish"] for r in results)})

asyncio.run(main())

