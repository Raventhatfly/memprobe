#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BAD_TEMPORAL_WORDS = re.compile(r"\b(earlier|previously|before|after|later|at some point)\b", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Use Qwen to rewrite MemProbe QA wording without changing facts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    tokenizer, model = _load_model(args.model, local_files_only=args.local_files_only)
    rewritten = []
    for idx, item in enumerate(_iter_jsonl(args.input)):
        if args.max_items is not None and idx >= args.max_items:
            break
        rewritten.append(_rewrite_item(item, tokenizer, model))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in rewritten:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"rewritten_items={len(rewritten)}")
    print(f"output={args.output}")


def _load_model(model_name: str, local_files_only: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.eval()
    return tokenizer, model


def _rewrite_item(item: dict, tokenizer, model) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite robot-memory multiple-choice questions. Preserve the exact answer, "
                "choice ids, canonical labels, and temporal anchor. Do not invent facts. "
                "Do not use vague temporal words such as earlier, previously, before, after, or later."
            ),
        },
        {"role": "user", "content": _prompt(item)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = tokenizer.batch_decode(output_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]
    payload = _extract_json(generated)

    new_item = dict(item)
    question = str(payload.get("question", item["question"])).strip()
    if not question or BAD_TEMPORAL_WORDS.search(question):
        question = item["question"]
    new_item["question"] = question

    choices_text = payload.get("choices", {})
    new_choices = []
    for choice in item["choices"]:
        updated = dict(choice)
        text = choices_text.get(choice["id"]) if isinstance(choices_text, dict) else None
        if isinstance(text, str) and text.strip():
            updated["text"] = text.strip()
        new_choices.append(updated)
    new_item["choices"] = new_choices
    new_item["wording_model"] = getattr(model, "name_or_path", "qwen")
    return new_item


def _prompt(item: dict) -> str:
    choices = {choice["id"]: choice["canonical"] for choice in item["choices"]}
    return json.dumps(
        {
            "task": "Rewrite only the natural-language surface form.",
            "question_family": item["question_family"],
            "current_question": item["question"],
            "structured_query": item["structured_query"],
            "answer_id": item["answer"],
            "choices_by_id": choices,
            "requirements": [
                "Return valid JSON only.",
                "Schema: {\"question\": string, \"choices\": {\"A\": string, \"B\": string, \"C\": string, \"D\": string}}.",
                "Keep the subtask index and query-frame anchor explicit.",
                "Do not use vague temporal words: earlier, previously, before, after, later.",
                "Do not mention canonical, filename, HDF5, target_id, provenance, or implementation details.",
                "Do not change answer id, choice ids, or canonical meaning.",
            ],
        },
        ensure_ascii=False,
    )


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
