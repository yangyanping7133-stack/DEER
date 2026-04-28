#!/usr/bin/env python3
"""
Plan A - Step 1: Inference only. Save full outputs for later judging.
"""
import asyncio, json, os, re, sys, time, statistics
from datetime import datetime
import httpx

API = "http://127.0.0.1:8000"
MODEL = "/root/models/Qwen/Qwen3-32B"
DATA_DIR = "/root/benchmarks/data/CRC-QAD/v1.1-pilot"
OUT_DIR = "/root/benchmarks/results/plan_a"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_TOKENS = 32768
TEMPERATURE = 0.6
CONCURRENCY = 3

CONFIDENCE_STOP = {"window": 20, "threshold": 0.95, "sustain": 30, "min_tokens": 2000}

PROMPT_FN = {
    "gsm8k": lambda q: q + "\nPlease solve this step by step.",
}


def parse_thinking(text):
    reasoning, content = "", text
    idx = text.find("<think")
    if idx >= 0:
        gt = text.find(">", idx)
        if gt >= 0:
            end = text.find("</think", gt)
            if end >= 0:
                close = text.find(">", end)
                reasoning = text[gt + 1:end].strip()
                content = text[close + 1:].strip() if close >= 0 else ""
            else:
                reasoning = text[gt + 1:].strip()
                content = ""
    return reasoning, content


async def stream_request(messages, method="baseline"):
    payload = {
        "model": MODEL, "messages": messages,
        "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if method == "m3":
        payload["confidence_stop"] = CONFIDENCE_STOP
    full_text = ""
    ttft = None
    t0 = time.perf_counter()
    usage = {}
    think_start = think_end = None
    in_think = False
    token_count = 0
    stop_reason = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
        async with client.stream("POST", f"{API}/v1/chat/completions",
                                  json=payload, headers={"Content-Type": "application/json"}) as resp:
            async for line in resp.aiter_lines():
                if not line.strip() or not line.startswith("data: "):
                    continue
                ds = line[6:]
                if ds.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(ds)
                except:
                    continue
                if data.get("usage"):
                    usage = data["usage"]
                choices = data.get("choices", [])
                if not choices:
                    continue
                ch = choices[0]
                delta = ch.get("delta", {})
                c = delta.get("content", "")
                if c:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    full_text += c
                    token_count += 1
                    if "<think" in full_text and not in_think:
                        gt = full_text.find(">", full_text.find("<think"))
                        if gt >= 0:
                            in_think = True
                            think_start = time.perf_counter() - t0
                    if "</think" in full_text and in_think:
                        in_think = False
                        think_end = time.perf_counter() - t0
                if ch.get("stop_reason"):
                    stop_reason = ch["stop_reason"]

    total_time = time.perf_counter() - t0
    if ttft is None:
        ttft = total_time
    reasoning, answer_content = parse_thinking(full_text)
    think_time = (think_end - think_start) if think_start and think_end else 0

    return {
        "total_time": round(total_time, 2),
        "ttft": round(ttft, 2),
        "thinking_time": round(think_time, 2),
        "completion_tokens": usage.get("completion_tokens", token_count),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "thinking_tokens_est": max(1, len(reasoning) // 4),
        "stop_reason": stop_reason,
        "answer_content": answer_content,
        "full_text": full_text,
    }


async def run_inference(dataset="gsm8k", method="baseline", data_file=None):
    if data_file:
        samples = json.load(open(data_file))
    else:
        samples = json.load(open(os.path.join(DATA_DIR, f"{dataset}.json")))
    prompt_fn = PROMPT_FN.get(dataset, lambda q: q)

    print(f"Plan A Step 1: {dataset}/{method}, {len(samples)} samples", flush=True)
    if method == "m3":
        print(f"  confidence_stop: {CONFIDENCE_STOP}", flush=True)
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}\n", flush=True)

    results = []
    for batch_start in range(0, len(samples), CONCURRENCY):
        batch = samples[batch_start:batch_start + CONCURRENCY]

        async def run_one(idx, sample):
            q = sample["question"]
            gt = sample.get("answer", "")
            msgs = [{"role": "user", "content": prompt_fn(q)}]
            r = await stream_request(msgs, method)
            r["index"] = idx
            r["id"] = sample.get("id", f"{dataset}-{idx}")
            r["question"] = q[:200]
            r["ground_truth"] = gt

            early_stopped = (r.get("stop_reason") == "confidence_stop")
            r["early_stopped"] = early_stopped
            if early_stopped and (not r["answer_content"] or len(r["answer_content"].strip()) <= 2):
                reasoning, _ = parse_thinking(r["full_text"])
                ti = r["full_text"].find("<think")
                if ti >= 0:
                    gi = r["full_text"].find(">", ti)
                    if gi >= 0:
                        reasoning = r["full_text"][gi + 1:]
                fu_answer, fu_time = await _followup_request(msgs, reasoning)
                r["answer_content"] = fu_answer
                r["followup_time"] = round(fu_time, 2)
                r["total_time"] = round(r["total_time"] + fu_time, 2)
            else:
                r["followup_time"] = 0.0
            return r

        tasks = [run_one(batch_start + j, s) for j, s in enumerate(batch)]
        out = await asyncio.gather(*tasks, return_exceptions=True)

        for r in out:
            if isinstance(r, Exception):
                results.append({"error": str(r), "index": -1})
            else:
                stopped_flag = "*" if r.get("early_stopped") else " "
                fu_info = f" fu={r['followup_time']:.1f}s" if r["followup_time"] > 0 else ""
                results.append(r)
                print(f"  [{r['index']}]{stopped_flag} E2E={r['total_time']:.1f}s tok={r['completion_tokens']} "
                      f"think_tok={r['thinking_tokens_est']} ttft={r['ttft']:.2f}s "
                      f"stop={r.get('stop_reason')}{fu_info}", flush=True)

    results.sort(key=lambda x: x.get("index", 999))

    save_path = os.path.join(OUT_DIR, f"{dataset}_{method}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    valid = [r for r in results if "error" not in r]
    stopped_count = sum(1 for r in valid if r.get("early_stopped"))
    print(f"\n{'='*50}")
    print(f"Done: {len(valid)} results, early_stopped={stopped_count}")
    print(f"Avg E2E: {statistics.mean([r['total_time'] for r in valid]):.1f}s")
    print(f"Avg TTFT: {statistics.mean([r['ttft'] for r in valid]):.2f}s")
    print(f"Avg Think tokens: {statistics.mean([r['thinking_tokens_est'] for r in valid]):.0f}")
    print(f"Avg Completion tokens: {statistics.mean([r['completion_tokens'] for r in valid]):.0f}")
    print(f"Saved: {save_path}")

    return results


async def _followup_request(messages, thinking_text):
    fmsg = [
        {"role": "user", "content": messages[0]["content"]},
        {"role": "assistant", "content": "<think >\n" + thinking_text + "\n</think >"},
        {"role": "user", "content": "/no_think\nBased ONLY on the reasoning above, output your final answer. Do NOT re-derive. Use the format: #### <answer> or \\boxed{<answer>}."},
    ]
    payload = {
        "model": MODEL, "messages": fmsg,
        "max_tokens": 512, "temperature": 0.6, "stream": False,
    }
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=30.0)) as client:
        resp = await client.post(f"{API}/v1/chat/completions",
                                  json=payload, headers={"Content-Type": "application/json"})
        data = resp.json()
    elapsed = time.perf_counter() - t0
    text = data["choices"][0]["message"]["content"]
    _, answer = parse_thinking(text)
    return answer.strip() if answer else text.strip(), elapsed


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
    method = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    data_file = sys.argv[3] if len(sys.argv) > 3 else None
    asyncio.run(run_inference(ds, method, data_file))
