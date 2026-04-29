#!/usr/bin/env python3
"""
Unified Benchmark Runner v2.0
Supports: GSM8K, MATH-500, AMC, GPQA, AIME, LiveCodeBench
Methods: baseline, deer
Features: 12 concurrency, checkpoint resume, structured JSON output
"""
import asyncio, json, math, os, re, sys, time, statistics
from datetime import datetime
from typing import Optional
import httpx

API = "http://127.0.0.1:8000"
MODEL = "/root/models/Qwen/Qwen3-32B"
DATA_DIR = "/root/benchmarks/data/CRC-QAD/v2.0-6sets"
PILOT_DIR = "/root/benchmarks/data/CRC-QAD/v2.0-6sets-pilot"
OUT_DIR = "/root/benchmarks/results/v2.0"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_TOKENS = 32768
TEMPERATURE = 0.6
CONCURRENCY = 12

DEER_PARAMS = {
    "threshold": 0.90,
    "prob_check_tokens": 10,
    "think_ratio": 0.6,
    "max_judge_steps": 2,
    "temperature": 0.6,
    "min_think_tokens": 300,
}

DATASET_CONFIG = {
    "gsm8k":         {"file": "gsm8k.json",         "type": "math", "prompt_suffix": "\nPlease solve this step by step."},
    "math500":       {"file": "math500.json",        "type": "math", "prompt_suffix": ""},
    "amc":           {"file": "amc.json",            "type": "math", "prompt_suffix": ""},
    "gpqa":          {"file": "gpqa.json",           "type": "mc",   "prompt_suffix": ""},
    "aime":          {"file": "aime.json",           "type": "math", "prompt_suffix": ""},
    "livecodebench": {"file": "livecodebench.json",  "type": "code", "prompt_suffix": "\nWrite a solution in Python."},
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


def apply_prompt(question, dataset_key):
    cfg = DATASET_CONFIG.get(dataset_key, {})
    suffix = cfg.get("prompt_suffix", "")
    return question + suffix if suffix else question


async def stream_request(messages):
    payload = {
        "model": MODEL, "messages": messages,
        "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
        "stream": True, "stream_options": {"include_usage": True},
    }
    full_text = ""
    ttft = None
    t0 = time.perf_counter()
    usage = {}
    think_start = think_end = None
    in_think = False
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
                except Exception:
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
                    if "<think" in full_text and not in_think:
                        gt = full_text.find(">", full_text.find("<think"))
                        if gt >= 0:
                            in_think = True
                            think_start = time.perf_counter() - t0
                    if "</think" in full_text and in_think:
                        in_think = False
                        think_end = time.perf_counter() - t0
                if ch.get("finish_reason"):
                    stop_reason = ch["finish_reason"]

    total_time = time.perf_counter() - t0
    if ttft is None:
        ttft = total_time
    reasoning, answer_content = parse_thinking(full_text)
    think_time = (think_end - think_start) if think_start and think_end else 0

    return {
        "total_time": round(total_time, 2),
        "ttft": round(ttft, 2),
        "thinking_time": round(think_time, 2),
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "thinking_tokens_est": max(1, len(reasoning) // 4),
        "stop_reason": stop_reason,
        "answer_content": answer_content,
        "full_text": full_text,
    }


async def api_call(messages, max_tokens, temperature=0.0, logprobs=False):
    payload = {
        "model": MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        "stream": False,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        resp = await client.post(
            f"{API}/v1/chat/completions",
            json=payload, headers={"Content-Type": "application/json"},
        )
        return resp.json()


async def deer_inference(question, dataset_key):
    question = apply_prompt(question, dataset_key)
    threshold = DEER_PARAMS["threshold"]
    prob_tokens = DEER_PARAMS["prob_check_tokens"]
    think_budget = int(DEER_PARAMS["think_ratio"] * MAX_TOKENS)
    max_steps = DEER_PARAMS["max_judge_steps"]
    temp = DEER_PARAMS["temperature"]
    min_think_tokens = DEER_PARAMS.get("min_think_tokens", 0)

    thinking = ""
    total_completion_tokens = 0
    prompt_tokens = 0
    judge_step = 0
    natural_end = False

    t0 = time.perf_counter()
    ttft = None
    think_start = None
    think_end = None
    deer_exited = False

    while judge_step < max_steps:
        if not thinking:
            messages = [{"role": "user", "content": question}]
        else:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": thinking},
            ]

        payload = {
            "model": MODEL, "messages": messages,
            "max_tokens": think_budget, "temperature": temp,
            "stream": True, "stream_options": {"include_usage": True},
            "stop": ["Wait"],
        }

        chunk = ""
        chunk_usage = {}
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
                    except Exception:
                        continue
                    if data.get("usage"):
                        chunk_usage = data["usage"]
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        c = delta.get("content", "")
                        if c:
                            chunk += c
                        if choices[0].get("finish_reason"):
                            stop_reason = choices[0]["finish_reason"]

        if ttft is None:
            ttft = time.perf_counter() - t0
        if think_start is None:
            think_start = time.perf_counter() - t0

        total_completion_tokens += chunk_usage.get("completion_tokens", 0)
        prompt_tokens = max(prompt_tokens, chunk_usage.get("prompt_tokens", 0))

        if "</think" in chunk:
            thinking += chunk
            think_end = time.perf_counter() - t0
            natural_end = True
            break

        thinking += chunk

        if stop_reason == "stop":
            thinking += "Wait"
        elif stop_reason != "length":
            think_end = time.perf_counter() - t0
            natural_end = True
            break

        judge_step += 1
        if judge_step >= max_steps:
            think_end = time.perf_counter() - t0
            break

        thinking_body = thinking
        idx_t = thinking_body.find("<think")
        if idx_t >= 0:
            gt = thinking_body.find(">", idx_t)
            if gt >= 0:
                thinking_body = thinking_body[gt + 1:]

        est_think = max(1, len(thinking_body) // 4)
        if est_think < min_think_tokens:
            thinking += "Wait"
            judge_step += 1
            if judge_step >= max_steps:
                think_end = time.perf_counter() - t0
                break
            continue

        prob_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": thinking_body + "\n</think >\n\n**Final Answer**\n\\boxed"},
        ]
        data = await api_call(prob_messages, prob_tokens, 0.0, logprobs=True)

        choice = data["choices"][0]
        token_probs = []
        logprobs_data = choice.get("logprobs") or {}
        if logprobs_data.get("content"):
            for t in logprobs_data["content"]:
                prob = math.exp(t["logprob"])
                token_probs.append(prob)

        confidence = geometric_mean(token_probs) if token_probs else 0.0

        if confidence >= threshold:
            deer_exited = True
            think_end = time.perf_counter() - t0
            break
        else:
            thinking += "Wait"

    if think_start and not think_end:
        think_end = time.perf_counter() - t0
    think_time = (think_end - think_start) if think_start and think_end else 0

    if natural_end:
        full_text = thinking
        reasoning, answer_content = parse_thinking(full_text)
    else:
        thinking_body = thinking
        idx_t = thinking_body.find("<think")
        if idx_t >= 0:
            gt = thinking_body.find(">", idx_t)
            if gt >= 0:
                thinking_body = thinking_body[gt + 1:]

        ans_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": thinking_body + "\n</think >\n"},
        ]
        answer_text, ans_usage, _ = await _stream_simple(ans_messages, 512, temp)
        total_completion_tokens += ans_usage.get("completion_tokens", 0)
        prompt_tokens = max(prompt_tokens, ans_usage.get("prompt_tokens", 0))

        full_text = thinking + "\n</think >\n" + answer_text
        reasoning, answer_content = parse_thinking(full_text)

    total_time = time.perf_counter() - t0

    return {
        "total_time": round(total_time, 2),
        "ttft": round(ttft, 2) if ttft else round(total_time, 2),
        "thinking_time": round(think_time, 2),
        "completion_tokens": total_completion_tokens,
        "prompt_tokens": prompt_tokens,
        "thinking_tokens_est": max(1, len(reasoning) // 4),
        "stop_reason": "natural" if natural_end else ("deer_exit" if deer_exited else "max_steps"),
        "answer_content": answer_content,
        "full_text": full_text,
        "deer_judge_steps": judge_step,
        "early_stopped": deer_exited,
    }


async def _stream_simple(messages, max_tokens, temperature):
    payload = {
        "model": MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        "stream": True, "stream_options": {"include_usage": True},
    }
    full_text = ""
    usage = {}
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


def load_checkpoint(save_path):
    if os.path.exists(save_path):
        try:
            results = json.load(open(save_path))
            done_ids = {r.get("id", r.get("index")) for r in results if r and "error" not in r}
            print(f"  Checkpoint: {len(done_ids)} existing results loaded", flush=True)
            return results, done_ids
        except Exception:
            pass
    return [], set()


async def run_dataset(dataset_key, method="baseline", pilot=True):
    cfg = DATASET_CONFIG[dataset_key]

    if pilot:
        data_path = os.path.join(PILOT_DIR, f"{dataset_key}_1.json")
    else:
        data_path = os.path.join(DATA_DIR, cfg["file"])

    samples = json.load(open(data_path))
    qtype = cfg["type"]

    save_path = os.path.join(OUT_DIR, f"{dataset_key}_{method}.json")
    results, done_ids = load_checkpoint(save_path)
    results_map = {r.get("id", r.get("index")): r for r in results if r}

    pending = [(i, s) for i, s in enumerate(samples) if s.get("id", i) not in done_ids]
    print(f"\n{'='*60}", flush=True)
    print(f"Dataset: {dataset_key} | Method: {method} | Samples: {len(samples)} | "
          f"Pending: {len(pending)} | Type: {qtype}", flush=True)
    print(f"Save: {save_path}", flush=True)
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}", flush=True)

    if not pending:
        print(f"  All done (from checkpoint).", flush=True)
        return results

    sem = asyncio.Semaphore(CONCURRENCY)

    async def _run(idx, sample):
        async with sem:
            q = sample["question"]
            gt = sample.get("answer", "")
            sid = sample.get("id", f"{dataset_key}-{idx}")

            if method == "deer":
                r = await deer_inference(q, dataset_key)
            else:
                q_prompted = apply_prompt(q, dataset_key)
                msgs = [{"role": "user", "content": q_prompted}]
                r = await stream_request(msgs)

            r["index"] = idx
            r["id"] = sid
            r["dataset"] = dataset_key
            r["question"] = q[:200]
            r["ground_truth"] = gt
            r["method"] = method
            r["qtype"] = qtype
            return r

    tasks = [_run(i, s) for i, s in pending]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    for item in completed:
        if isinstance(item, Exception):
            print(f"  ERROR: {item}", flush=True)
            results.append({"error": str(item), "index": -1})
        else:
            sid = item.get("id", item.get("index"))
            results_map[sid] = item
            flag = "*" if item.get("early_stopped") else " "
            print(
                f"  [{item['index']}]{flag} E2E={item['total_time']:.1f}s "
                f"tok={item['completion_tokens']} think_tok={item['thinking_tokens_est']} "
                f"steps={item.get('deer_judge_steps', '-')} stop={item.get('stop_reason')}",
                flush=True,
            )

    results = sorted(results_map.values(), key=lambda x: x.get("index", 999))
    with open(save_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    valid = [r for r in results if "error" not in r]
    print(f"\n  Done: {len(valid)}/{len(samples)}", flush=True)
    if valid:
        print(f"  Avg E2E: {statistics.mean(r['total_time'] for r in valid):.1f}s", flush=True)
        print(f"  Avg TTFT: {statistics.mean(r['ttft'] for r in valid):.2f}s", flush=True)
        print(f"  Avg Think tokens: {statistics.mean(r['thinking_tokens_est'] for r in valid):.0f}", flush=True)
        print(f"  Avg Completion tokens: {statistics.mean(r['completion_tokens'] for r in valid):.0f}", flush=True)
        if method == "deer":
            print(f"  Avg DEER steps: {statistics.mean(r['deer_judge_steps'] for r in valid):.1f}", flush=True)

    return results


async def run_judge(dataset_key, method="baseline"):
    cfg = DATASET_CONFIG[dataset_key]
    qtype = cfg["type"]

    in_path = os.path.join(OUT_DIR, f"{dataset_key}_{method}.json")
    results = json.load(open(in_path))
    save_path = os.path.join(OUT_DIR, f"{dataset_key}_{method}_judged.json")

    JUDGE_SYSTEM = """你是一个严格的答案评判裁判模型。判断被评测模型的输出是否与标准答案一致。

【评判规则】
1. 数学题：数值在数学上等价或非常接近（误差<1%）则正确
2. 选择题：选项一致则正确
3. 代码题：核心算法逻辑正确则正确
4. 只看最终答案，不看推理过程
5. 只输出一行JSON：{"correct": true} 或 {"correct": false}"""

    JUDGE_USER = """/no_think
【题目类型】{qtype}
【标准答案】{ground_truth}
【被评测模型的输出】
{model_output}

请判断模型输出是否正确。只输出一行JSON。"""

    def parse_judge(content):
        answer = content
        te = content.find("</think")
        if te >= 0:
            cg = content.find(">", te)
            if cg >= 0:
                answer = content[cg + 1:].strip()
        json_match = re.search(r'\{[^{}]*"correct"\s*:\s*(true|false)[^{}]*\}', answer, re.IGNORECASE)
        if json_match:
            return json.loads(json_match.group()).get("correct")
        if re.search(r'"correct"\s*:\s*true', answer, re.IGNORECASE):
            return True
        if re.search(r'"correct"\s*:\s*false', answer, re.IGNORECASE):
            return False
        return None

    sem = asyncio.Semaphore(4)

    async def judge_one(r):
        if "error" in r:
            return r.get("id", r.get("index")), None, 0, "HAS_ERROR"
        async with sem:
            prompt = JUDGE_USER.format(
                qtype=qtype,
                ground_truth=r.get("ground_truth", "")[:500],
                model_output=(r.get("answer_content", "") or "")[:2000],
            )
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 128, "temperature": 0, "stream": False,
            }
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                resp = await client.post(f"{API}/v1/chat/completions",
                                          headers={"Content-Type": "application/json"}, json=payload)
                data = resp.json()
            elapsed = time.perf_counter() - t0
            content = data["choices"][0]["message"]["content"]
            correct = parse_judge(content)
            return r.get("id", r.get("index")), correct, round(elapsed, 2), content

    print(f"\nJudging: {dataset_key}/{method}", flush=True)
    tasks = [judge_one(r) for r in results if "error" not in r]
    judged = await asyncio.gather(*tasks, return_exceptions=True)

    results_map = {r.get("id", r.get("index")): r for r in results}
    for item in judged:
        if isinstance(item, Exception):
            continue
        sid, correct, judge_time, raw = item
        if sid in results_map:
            results_map[sid]["judge_correct"] = correct
            results_map[sid]["judge_time"] = judge_time
            results_map[sid]["judge_raw"] = raw[:200] if isinstance(raw, str) else str(raw)[:200]

    results = sorted(results_map.values(), key=lambda x: x.get("index", 999))
    with open(save_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    valid = [r for r in results if "error" not in r and r.get("judge_correct") is not None]
    correct_count = sum(1 for r in valid if r["judge_correct"])
    acc = correct_count / len(valid) * 100 if valid else 0
    print(f"  Accuracy: {correct_count}/{len(valid)} = {acc:.0f}%", flush=True)
    return results


def generate_report(all_results):
    report_lines = []
    report_lines.append("# DEER CoT Compression Benchmark Report v2.0")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n## Environment")
    report_lines.append(f"- Model: Qwen3-32B")
    report_lines.append(f"- Framework: vLLM-Ascend (8x Ascend 910B4)")
    report_lines.append(f"- API: {API}")
    report_lines.append(f"- Temperature: {TEMPERATURE}")
    report_lines.append(f"- Max Tokens: {MAX_TOKENS}")
    report_lines.append(f"- Concurrency: {CONCURRENCY}")
    report_lines.append(f"\n## DEER Parameters")
    for k, v in DEER_PARAMS.items():
        report_lines.append(f"- {k}: {v}")

    report_lines.append(f"\n## Results Summary\n")
    report_lines.append(f"| Dataset | Method | Accuracy | Avg E2E (s) | Avg TTFT (s) | Think Tokens | Completion Tokens | DEER Steps | Speedup |")
    report_lines.append(f"|---------|--------|----------|-------------|--------------|-------------|-------------------|------------|---------|")

    summary = {}
    for dataset_key, methods in all_results.items():
        for method, res_list in methods.items():
            valid = [r for r in res_list if "error" not in r]
            if not valid:
                continue
            judged = [r for r in valid if r.get("judge_correct") is not None]
            correct = sum(1 for r in judged if r["judge_correct"]) if judged else 0
            total_j = len(judged) if judged else 0
            acc = f"{correct}/{total_j}" if total_j else "N/A"
            avg_e2e = statistics.mean(r["total_time"] for r in valid)
            avg_ttft = statistics.mean(r["ttft"] for r in valid)
            avg_think = statistics.mean(r["thinking_tokens_est"] for r in valid)
            avg_tok = statistics.mean(r["completion_tokens"] for r in valid)
            avg_steps = statistics.mean(r.get("deer_judge_steps", 0) for r in valid)

            key = (dataset_key, method)
            summary[key] = {"acc": acc, "avg_e2e": avg_e2e, "avg_ttft": avg_ttft,
                           "avg_think": avg_think, "avg_tok": avg_tok, "avg_steps": avg_steps,
                           "correct": correct, "total_j": total_j, "n": len(valid)}
            report_lines.append(
                f"| {dataset_key} | {method} | {acc} | {avg_e2e:.1f} | {avg_ttft:.2f} | "
                f"{avg_think:.0f} | {avg_tok:.0f} | {avg_steps:.1f} | - |"
            )

    speedup_rows = []
    for ds in DATASET_CONFIG:
        bl = summary.get((ds, "baseline"))
        dr = summary.get((ds, "deer"))
        if bl and dr:
            tok_speedup = (1 - dr["avg_tok"] / bl["avg_tok"]) * 100 if bl["avg_tok"] > 0 else 0
            time_speedup = (1 - dr["avg_e2e"] / bl["avg_e2e"]) * 100 if bl["avg_e2e"] > 0 else 0
            think_reduction = (1 - dr["avg_think"] / bl["avg_think"]) * 100 if bl["avg_think"] > 0 else 0
            speedup_rows.append({
                "dataset": ds,
                "bl_acc": bl["acc"], "dr_acc": dr["acc"],
                "tok_speedup": tok_speedup, "time_speedup": time_speedup,
                "think_reduction": think_reduction,
                "bl_e2e": bl["avg_e2e"], "dr_e2e": dr["avg_e2e"],
                "bl_tok": bl["avg_tok"], "dr_tok": dr["avg_tok"],
                "bl_think": bl["avg_think"], "dr_think": dr["avg_think"],
                "avg_steps": dr["avg_steps"],
            })

    if speedup_rows:
        report_lines.append(f"\n## Comparison: Baseline vs DEER\n")
        report_lines.append(f"| Dataset | BL Acc | DEER Acc | Time Speedup | Token Reduction | Think Reduction | DEER Steps |")
        report_lines.append(f"|---------|--------|----------|-------------|-----------------|-----------------|------------|")
        for r in speedup_rows:
            report_lines.append(
                f"| {r['dataset']} | {r['bl_acc']} | {r['dr_acc']} | "
                f"{r['time_speedup']:.1f}% | {r['tok_speedup']:.1f}% | "
                f"{r['think_reduction']:.1f}% | {r['avg_steps']:.1f} |"
            )

        avg_time_speedup = statistics.mean(r["time_speedup"] for r in speedup_rows)
        avg_tok_speedup = statistics.mean(r["tok_speedup"] for r in speedup_rows)
        avg_think_red = statistics.mean(r["think_reduction"] for r in speedup_rows)

        report_lines.append(f"\n## Overall Averages")
        report_lines.append(f"- Avg Time Speedup: {avg_time_speedup:.1f}%")
        report_lines.append(f"- Avg Token Reduction: {avg_tok_speedup:.1f}%")
        report_lines.append(f"- Avg Think Token Reduction: {avg_think_red:.1f}%")

        target_met = avg_time_speedup >= 20
        report_lines.append(f"\n## Conclusion")
        report_lines.append(f"- Target (speedup >= 20%): {'**MET**' if target_met else '**NOT MET**'}")
        if not target_met:
            report_lines.append(f"- Gap: {20 - avg_time_speedup:.1f}% additional speedup needed")
            report_lines.append(f"- Recommendation: Consider lowering threshold, increasing max_judge_steps, or adjusting think_ratio")

    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    report_json = {
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "model": "Qwen3-32B",
            "framework": "vLLM-Ascend",
            "hardware": "8x Ascend 910B4",
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "concurrency": CONCURRENCY,
        },
        "deer_params": DEER_PARAMS,
        "summary": {f"{k[0]}_{k[1]}": v for k, v in summary.items()},
        "comparison": speedup_rows,
    }
    json_path = os.path.join(OUT_DIR, "report.json")
    with open(json_path, "w") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved: {report_path}")
    print(f"Report JSON: {json_path}")
    return report_path


async def main():
    datasets = sys.argv[1].split(",") if len(sys.argv) > 1 else list(DATASET_CONFIG.keys())
    method = sys.argv[2] if len(sys.argv) > 2 else "all"
    skip_judge = "--skip-judge" in sys.argv
    skip_report = "--skip-report" in sys.argv
    pilot = "--full" not in sys.argv

    methods = ["baseline", "deer"] if method == "all" else [method]

    if pilot:
        print(f"Pilot mode: 1 question per dataset", flush=True)
    else:
        print(f"Full mode: all questions", flush=True)

    all_results = {}
    for ds in datasets:
        if ds not in DATASET_CONFIG:
            print(f"Unknown dataset: {ds}, skipping")
            continue
        all_results[ds] = {}
        for m in methods:
            res = await run_dataset(ds, m, pilot=pilot)
            all_results[ds][m] = res

            if not skip_judge:
                res_judged = await run_judge(ds, m)
                all_results[ds][m] = res_judged

    if not skip_report:
        generate_report(all_results)


if __name__ == "__main__":
    asyncio.run(main())
