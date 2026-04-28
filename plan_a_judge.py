#!/usr/bin/env python3
"""
Plan A - Step 2: Judge accuracy using local Qwen3-32B as referee.
Reads inference results, sends to local model for judging.
"""
import asyncio, json, os, re, sys, time, statistics
from datetime import datetime
import httpx

API = "http://127.0.0.1:8000"
MODEL = "/root/models/Qwen/Qwen3-32B"
IN_DIR = "/root/benchmarks/results/plan_a"
OUT_DIR = "/root/benchmarks/results/plan_a"
CONCURRENCY = 3

JUDGE_SYSTEM = """你是一个严格的答案评判裁判模型（Judge Model）。你的任务是判断被评测模型的输出是否与标准答案一致。

【评判规则】
1. 数学题：如果模型输出的数值与标准答案的数值在数学上等价或非常接近（绝对误差<1%或相对误差<1%），则判为正确
2. 选择题：如果模型选择的选项与标准答案一致，则判为正确
3. QA题：如果模型输出的关键信息与标准答案的核心内容匹配，则判为正确
4. 只看最终答案，不看推理过程
5. 必须只输出一行JSON：{"correct": true} 或 {"correct": false}"""

JUDGE_USER = """/no_think
【题目类型】{qtype}
【标准答案】{ground_truth}
【被评测模型的输出（不含思考过程）】
{model_output}

请判断模型输出是否正确。只输出一行JSON。"""


def parse_judge_content(content):
    answer = content
    te = content.find("</think")
    if te >= 0:
        cg = content.find(">", te)
        if cg >= 0:
            answer = content[cg + 1:].strip()

    json_match = re.search(r'\{[^{}]*"correct"\s*:\s*(true|false)[^{}]*\}', answer, re.IGNORECASE)
    if json_match:
        result = json.loads(json_match.group())
        return result.get("correct")

    if re.search(r'"correct"\s*:\s*true', answer, re.IGNORECASE):
        return True
    if re.search(r'"correct"\s*:\s*false', answer, re.IGNORECASE):
        return False
    return None


async def judge_one(ground_truth, model_output, qtype="math"):
    prompt = JUDGE_USER.format(
        qtype=qtype,
        ground_truth=ground_truth[:500],
        model_output=(model_output or "NO OUTPUT")[:2000],
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 128,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        resp = await client.post(f"{API}/v1/chat/completions",
                                  headers={"Content-Type": "application/json"},
                                  json=payload)
        data = resp.json()
    elapsed = time.perf_counter() - t0

    content = data["choices"][0]["message"]["content"]
    correct = parse_judge_content(content)
    return correct, round(elapsed, 2), content


async def run_judge(dataset="gsm8k", method="baseline", qtype="math"):
    in_path = os.path.join(IN_DIR, f"{dataset}_{method}.json")
    results = json.load(open(in_path))

    print(f"Plan A Step 2: Judging {dataset}/{method}, {len(results)} results", flush=True)
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}\n", flush=True)

    t_start = time.time()

    for batch_start in range(0, len(results), CONCURRENCY):
        batch = results[batch_start:batch_start + CONCURRENCY]

        async def judge_item(r):
            if "error" in r:
                return r["index"], None, 0, "HAS_ERROR"
            model_output = r.get("answer_content", "")
            gt = r.get("ground_truth", "")
            correct, judge_time, raw = await judge_one(gt, model_output, qtype)
            return r["index"], correct, judge_time, raw

        tasks = [judge_item(r) for r in batch]
        out = await asyncio.gather(*tasks, return_exceptions=True)

        for item in out:
            if isinstance(item, Exception):
                print(f"  JUDGE ERROR: {item}", flush=True)
                continue
            idx, correct, judge_time, raw = item
            results[idx]["judge_correct"] = correct
            results[idx]["judge_time"] = judge_time
            results[idx]["judge_raw"] = raw[:200] if isinstance(raw, str) else str(raw)[:200]
            print(f"  [{idx}] correct={correct} judge_time={judge_time:.1f}s", flush=True)

    save_path = os.path.join(OUT_DIR, f"{dataset}_{method}_judged.json")
    with open(save_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    valid = [r for r in results if "error" not in r]
    judged = [r for r in valid if r.get("judge_correct") is not None]
    correct_count = sum(1 for r in judged if r["judge_correct"])
    acc = correct_count / len(judged) * 100 if judged else 0

    avg_e2e = statistics.mean([r["total_time"] for r in valid])
    avg_ttft = statistics.mean([r["ttft"] for r in valid])
    avg_think = statistics.mean([r["thinking_time"] for r in valid])
    avg_tok = statistics.mean([r["completion_tokens"] for r in valid])
    avg_judge = statistics.mean([r["judge_time"] for r in judged]) if judged else 0

    print(f"\n{'='*50}")
    print(f"JUDGE RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy: {correct_count}/{len(judged)} = {acc:.0f}%")
    print(f"  Avg E2E: {avg_e2e:.1f}s | TTFT: {avg_ttft:.2f}s | Think: {avg_think:.1f}s | Tokens: {avg_tok:.0f}")
    print(f"  Avg Judge time: {avg_judge:.1f}s")
    print(f"  Total judge wall time: {(time.time()-t_start):.1f}s")
    print(f"  Saved: {save_path}")

    for i, r in enumerate(valid):
        status = "CORRECT" if r.get("judge_correct") else "WRONG"
        gt_num = r.get("ground_truth", "")[:80]
        ans_preview = (r.get("answer_content", "") or "")[:80]
        print(f"\n  [{i}] {status}")
        print(f"    GT: {gt_num}")
        print(f"    Model: {ans_preview}")
        print(f"    Judge time: {r.get('judge_time', 0):.1f}s")
        if r.get("judge_raw"):
            print(f"    Judge raw: {r['judge_raw'][:150]}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
    method = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    qtype = sys.argv[3] if len(sys.argv) > 3 else "math"
    asyncio.run(run_judge(ds, method, qtype))
