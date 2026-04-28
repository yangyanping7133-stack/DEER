#!/usr/bin/env python3
"""
DEER (Dynamic Early Exit in Reasoning) - Qwen3-32B via vLLM OpenAI API

Multi-step state machine:
  Think -> ProbCheck -> Think/Answer
  At each "Wait" transition, probe model confidence via short trial answer.
  Exit thinking when geometric_mean(token_probs) >= threshold AND last token is </think tag.

Adapted from: https://github.com/iie-ycx/DEER (vllm-deer-qwen3.py)
"""
import json, math, os, re, sys, time, statistics
from datetime import datetime
import httpx

API = "http://127.0.0.1:8000"
MODEL = "/root/models/Qwen/Qwen3-32B"
DATA_DIR = "/root/benchmarks/data/CRC-QAD/v1.1-pilot"
OUT_DIR = "/root/benchmarks/results/plan_a"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_TOKENS = 32768
TEMPERATURE = 0.6
THRESHOLD = 0.95
MAX_JUDGE_STEPS = 3
THINK_RATIO = 0.8
PROB_CHECK_TOKENS = 20

PROMPT_FN = {
    "gsm8k": lambda q: q + "\nPlease solve this step by step.",
}


def geometric_mean(probs):
    if not probs:
        return 0.0
    return math.exp(sum(math.log(max(p, 1e-10)) for p in probs) / len(probs))


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


def _api_call(messages, max_tokens, temperature=0.0, stop=None, logprobs=False, stream=False):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stop:
        payload["stop"] = stop
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1
    if stream:
        payload["stream_options"] = {"include_usage": True}

    timeout = httpx.Timeout(1800.0 if stream else 600.0, connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{API}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        return resp.json()


def _api_stream(messages, max_tokens, temperature=0.0, stop=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if stop:
        payload["stop"] = stop

    full_text = ""
    usage = {}
    stop_reason = None
    timeout = httpx.Timeout(1800.0, connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST", f"{API}/v1/chat/completions",
            json=payload, headers={"Content-Type": "application/json"},
        ) as resp:
            for line in resp.iter_lines():
                if not line.strip() or not line.startswith("data: "):
                    continue
                ds = line[6:]
                if ds.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(ds)
                except Exception:
                    continue
                if data.get("usage"):
                    usage = data["usage"]
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    c = delta.get("content", "")
                    if c:
                        full_text += c
                    if choices[0].get("finish_reason"):
                        stop_reason = choices[0]["finish_reason"]
    return full_text, usage, stop_reason


def deer_inference(question, prompt_fn=None, verbose=False):
    if prompt_fn:
        question = prompt_fn(question)

    thinking = ""
    total_completion_tokens = 0
    prompt_tokens = 0
    judge_step = 0
    think_budget = int(THINK_RATIO * MAX_TOKENS)

    t0 = time.perf_counter()
    ttft = None
    think_start = None
    think_end = None
    deer_exited = False

    while judge_step < MAX_JUDGE_STEPS:
        if not thinking:
            messages = [{"role": "user", "content": question}]
        else:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": thinking},
            ]

        chunk, usage, stop_reason = _api_stream(
            messages, think_budget, TEMPERATURE, stop=["Wait"]
        )

        if ttft is None:
            ttft = time.perf_counter() - t0
        if think_start is None:
            think_start = time.perf_counter() - t0

        total_completion_tokens += usage.get("completion_tokens", 0)
        prompt_tokens = max(prompt_tokens, usage.get("prompt_tokens", 0))

        if "</think" in chunk:
            think_part = chunk.split("</think")[0]
            thinking += think_part
            think_end = time.perf_counter() - t0
            if verbose:
                print(f"    [Think #{judge_step+1}] Natural end, +{len(think_part)} chars", flush=True)
            break

        thinking += chunk
        if verbose:
            print(
                f"    [Think #{judge_step+1}] +{len(chunk)} chars, stop={stop_reason}",
                flush=True,
            )

        if stop_reason == "stop":
            thinking += "Wait"
            do_prob = True
        elif stop_reason == "length":
            do_prob = True
        else:
            think_end = time.perf_counter() - t0
            break

        judge_step += 1
        if judge_step >= MAX_JUDGE_STEPS:
            think_end = time.perf_counter() - t0
            break

        if not do_prob:
            break

        thinking_body = thinking
        idx_t = thinking_body.find("<think")
        if idx_t >= 0:
            gt = thinking_body.find(">", idx_t)
            if gt >= 0:
                thinking_body = thinking_body[gt + 1:]

        prob_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": thinking_body + "\n</think >\n\n**Final Answer**\n\\boxed"},
        ]
        data = _api_call(
            prob_messages, PROB_CHECK_TOKENS, 0.0,
            logprobs=True, stream=False,
        )

        choice = data["choices"][0]
        token_probs = []

        logprobs_data = choice.get("logprobs") or {}
        if logprobs_data.get("content"):
            for t in logprobs_data["content"]:
                prob = math.exp(t["logprob"])
                token_probs.append(prob)

        confidence = geometric_mean(token_probs) if token_probs else 0.0

        if verbose:
            print(
                f"    [ProbCheck #{judge_step}] conf={confidence:.4f}, "
                f"n_tokens={len(token_probs)}",
                flush=True,
            )

        if confidence >= THRESHOLD:
            deer_exited = True
            think_end = time.perf_counter() - t0
            if verbose:
                print(f"    [DEER] Confident ({confidence:.4f})! Exiting think phase.", flush=True)
            break
        else:
            thinking += "Wait"
            if verbose:
                print(f"    [DEER] Low conf ({confidence:.4f}), continue thinking...", flush=True)

    if think_start and not think_end:
        think_end = time.perf_counter() - t0
    think_time = (think_end - think_start) if think_start and think_end else 0

    thinking_body = thinking
    idx_t = thinking_body.find("<think")
    if idx_t >= 0:
        gt = thinking_body.find(">", idx_t)
        if gt >= 0:
            thinking_body = thinking_body[gt + 1:]

    ans_messages = [
        {"role": "user", "content": "/no_think " + question},
        {"role": "assistant", "content": thinking_body + "\n</think >\n\n**Final Answer**\n\\boxed"},
    ]
    answer_text, ans_usage, _ = _api_stream(ans_messages, MAX_TOKENS, TEMPERATURE)
    total_completion_tokens += ans_usage.get("completion_tokens", 0)
    prompt_tokens = max(prompt_tokens, ans_usage.get("prompt_tokens", 0))

    total_time = time.perf_counter() - t0
    full_text = thinking + "\n</think >\n\n**Final Answer**\n\\boxed" + answer_text
    reasoning, answer_content = parse_thinking(full_text)

    return {
        "total_time": round(total_time, 2),
        "ttft": round(ttft, 2) if ttft else round(total_time, 2),
        "thinking_time": round(think_time, 2),
        "completion_tokens": total_completion_tokens,
        "prompt_tokens": prompt_tokens,
        "thinking_tokens_est": max(1, len(reasoning) // 4),
        "stop_reason": "deer_exit" if deer_exited else "natural",
        "answer_content": answer_content,
        "full_text": full_text,
        "deer_judge_steps": judge_step,
        "early_stopped": deer_exited or judge_step > 0,
    }


def _run_one(idx, sample, prompt_fn, verbose):
    q = sample["question"]
    gt = sample.get("answer", "")
    sid = sample.get("id", f"gsm8k-{idx}")
    print(f"[{idx}] start: {q[:60]}...", flush=True)
    r = deer_inference(q, prompt_fn, verbose=verbose)
    r["index"] = idx
    r["id"] = sid
    r["question"] = q[:200]
    r["ground_truth"] = gt
    flag = "*" if r["early_stopped"] else " "
    print(
        f"  [{idx}]{flag} E2E={r['total_time']:.1f}s tok={r['completion_tokens']} "
        f"think_tok={r['thinking_tokens_est']} steps={r['deer_judge_steps']} "
        f"stop={r['stop_reason']}",
        flush=True,
    )
    return r


def main():
    import concurrent.futures
    ds = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
    data_file = sys.argv[2] if len(sys.argv) > 2 else None
    verbose = "--verbose" in sys.argv
    concurrency = 16

    if data_file:
        samples = json.load(open(data_file))
    else:
        samples = json.load(open(os.path.join(DATA_DIR, f"{ds}.json")))

    prompt_fn = PROMPT_FN.get(ds, lambda q: q)

    print(f"DEER Inference: {ds}, {len(samples)} samples, concurrency={concurrency}", flush=True)
    print(f"  threshold={THRESHOLD}, prob_tokens={PROB_CHECK_TOKENS}, "
          f"think_ratio={THINK_RATIO}, temp={TEMPERATURE}", flush=True)
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}\n", flush=True)

    save_path = os.path.join(OUT_DIR, f"{ds}_deer.json")
    t_start = time.time()
    results = [None] * len(samples)
    done_count = 0

    def _save():
        with open(save_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_run_one, i, s, prompt_fn, verbose): i
            for i, s in enumerate(samples)
        }
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                err_str = str(e)
                is_oom = "oom" in err_str.lower() or "out of memory" in err_str.lower() or "CUDA" in err_str or "memory" in err_str.lower()
                tag = "OOM" if is_oom else "ERROR"
                print(f"  [{idx}] {tag}: {err_str[:200]}", flush=True)
                results[idx] = {"index": idx, "error": err_str, "oom": is_oom}
            done_count += 1
            _save()
            print(f"  >> saved {done_count}/{len(samples)}", flush=True)

    valid = [r for r in results if r and "error" not in r]
    stopped = sum(1 for r in valid if r.get("early_stopped"))
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done: {len(valid)} results in {elapsed:.1f}s ({elapsed/60:.1f}min), early_stopped={stopped}")
    if valid:
        print(f"Avg E2E: {statistics.mean([r['total_time'] for r in valid]):.1f}s")
        print(f"Avg TTFT: {statistics.mean([r['ttft'] for r in valid]):.2f}s")
        print(f"Avg Think tokens: {statistics.mean([r['thinking_tokens_est'] for r in valid]):.0f}")
        print(f"Avg Completion tokens: {statistics.mean([r['completion_tokens'] for r in valid]):.0f}")
        print(f"Avg DEER steps: {statistics.mean([r['deer_judge_steps'] for r in valid]):.1f}")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
