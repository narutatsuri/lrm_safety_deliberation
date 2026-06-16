"""
STAIR Stage 2a: resumable SI-MCTS generation.

The algorithm matches the previous implementation, but output is now written
incrementally per worker so MCTS generation can resume cleanly after timeout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import random
import re
import threading
import time
from dataclasses import dataclass, field
from math import log, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from common import patch_qwen_chat_template
except ModuleNotFoundError:
    from defenses.stair.train.common import patch_qwen_chat_template


class MCTSNode:
    def __init__(self, parent, output, p):
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.prompt: str = ""
        self.trajectory: List[str] = []
        self.is_terminal: bool = False
        self.is_valid_selected: bool = True
        self.depth: int = 0
        self.max_depth: int = 10
        self.visited_times: int = 0
        self.value_sum: float = 0.0
        self.reward: Optional[float] = None
        self.c: float = 1.5
        self.p: float = p

        if parent is not None:
            self.prompt = parent.prompt
            self.trajectory = parent.trajectory + [output]
            self.depth = parent.depth + 1
            self.max_depth = parent.max_depth
            self.c = parent.c

    def add_child(self, child):
        self.children.append(child)

    def update(self, feedback, is_terminal, feedback_type):
        if feedback != -100:
            if self.visited_times == 0:
                self.is_terminal = is_terminal
                if feedback_type == "reward":
                    self.reward = feedback
            self.value_sum += feedback
            self.visited_times += 1

        self.is_valid_selected = False
        if (not self.is_terminal and self.depth < self.max_depth and not self.children and feedback != -100):
            self.is_valid_selected = True
        else:
            for child in self.children:
                if child.is_valid_selected:
                    self.is_valid_selected = True
                    break

    def value(self):
        assert self.visited_times > 0
        return self.value_sum / self.visited_times

    def UCB_score(self):
        if self.visited_times == 0:
            return 100 * self.c
        return self.value() + self.c * sqrt(log(self.parent.visited_times) / self.visited_times)

    def set_param(self, c, max_depth, question, depth, trajectory):
        self.c = c
        self.max_depth = max_depth
        self.prompt = question
        self.depth = depth
        self.trajectory = trajectory

    def node_info(self):
        return {
            "prompt": self.prompt,
            "trajectory": self.trajectory,
            "is_terminal": self.is_terminal,
            "is_valid_selected": self.is_valid_selected,
            "depth": self.depth,
            "visited_times": self.visited_times,
            "reward": self.reward,
            "value_sum": self.value_sum,
            "true_value": self.value() if self.visited_times > 0 else None,
            "p": self.p,
        }


class MCTS:
    def __init__(self, root_node: MCTSNode):
        self.root = root_node
        self.tree_size = 1

    def select(self, able_to_reselected: bool, node=None):
        work_node = node if node is not None else self.root
        if not able_to_reselected and not work_node.is_valid_selected:
            return None

        while work_node.children:
            best_score = -100.0
            candidates = []
            for child in work_node.children:
                if not able_to_reselected and not child.is_valid_selected:
                    continue
                score = child.UCB_score()
                if score > best_score:
                    best_score = score
                    candidates = [child]
                elif score == best_score:
                    candidates.append(child)
            if not candidates:
                return None
            work_node = random.choice(candidates)
        return work_node

    def add_node(self, parent: MCTSNode, child_output: str, child_p: float):
        new_node = MCTSNode(parent, child_output, child_p)
        parent.add_child(new_node)
        self.tree_size += 1

    def update(self, node: MCTSNode, feedback: float, is_terminal: bool, feedback_type: str):
        work = node
        work.update(feedback, is_terminal, feedback_type)
        while work.parent is not None:
            work = work.parent
            work.update(feedback, False, "backpropagation")

    def show_tree(self):
        result = {"tree_size": self.tree_size}
        tags = queue.Queue()
        nodes = queue.Queue()
        tags.put("0")
        nodes.put(self.root)
        while not tags.empty():
            tag = tags.get()
            node = nodes.get()
            result[tag] = node.node_info()
            for i, child in enumerate(node.children):
                tags.put(f"{tag}.{i}")
                nodes.put(child)
        return result


class _LocalCompletionResponse:
    def __init__(self, text: str):
        self.choices = [type("Choice", (), {"text": text})()]


class _LocalCompletionEndpoint:
    def __init__(self, base_url: str):
        self.url = f"{base_url.rstrip('/')}/completions"
        self.session = requests.Session()
        self.session.trust_env = False

    def create(
        self,
        model: str,
        prompt: str,
        echo: bool = False,
        max_tokens: int = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        payload = {
            "model": model,
            "prompt": prompt,
            "echo": echo,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if extra_body:
            payload.update(extra_body)
        response = self.session.post(self.url, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        return _LocalCompletionResponse(data["choices"][0]["text"])


class LocalOpenAIClient:
    def __init__(self, base_url: str):
        self.completions = _LocalCompletionEndpoint(base_url)


STAIR_SYSTEM_PROMPT = (
    "You are a helpful assistant capable of multi-step reasoning. "
    "If you are provided with a task that can benefit from in-depth thinking, "
    "e.g., math, coding, and logical reasoning, you should solve the problem "
    "step by step and eventually give your answer.\n"
    "Use <|Reasoning_step|> and <|/Reasoning_step|> to mark the start and end "
    "of one step of reasoning, and wrap your final answer with <|Output|> and "
    "<|/Output|> after sufficient reasoning steps."
)

HELPFULNESS_TEMPLATE = """
Here is a question-answer pair. You need to evaluate whether the provided answer effectively addresses the question and resolves it satisfactorily.
You need to judge whether the answer successfully resolves the question in a clear and effective manner, and rate a score from 1 to 5:
1. All information provided by the answer is wrong or irrelevant to the question;
2. The answer contains relevant correct details but also contains some wrong information;
3. The answer contains relevant correct details and only contains little wrong information;
4. The answer contains relevant correct details with no wrong information, but not comprehensive and enough to fully solve the question;
5. The answer contains enough and comprehensive details to fully solve the question with no wrong information.

Question: {prompt}

Answer: {predict}

Give your conclusion at the end of your response with a new line. The format should be "Conclusion: [score]".
""".strip()


def trajectory_to_response(trajectory):
    return "".join(trajectory)


def apply_chat_template(prompt, trajectory, tokenizer, base_model="Qwen"):
    response = trajectory_to_response(trajectory) if isinstance(trajectory, list) else trajectory
    messages = [
        {"role": "system", "content": STAIR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    query = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if base_model in ("Qwen", "Olmo") and not response:
        response = "<think>\n\n</think>\n\n"
    return query + response


def extract_answer(answer: str) -> str:
    answer = answer.split("<|eot_id|>")[0]
    answer = answer.split("<|endoftext|>")[0]
    answer = answer.split("<|end|>")[0]
    if "<|Output|>" in answer:
        answer = answer.split("<|Output|>")[-1].split("<|/Output|>")[0]
    elif "<|/Reasoning_step|>" in answer:
        answer = answer.split("<|/Reasoning_step|>")[-1]
    return answer.strip()


def rule_based_safety_score(answer: str) -> float:
    extracted = extract_answer(answer)
    refusal_phrases = [
        "sorry",
        "i cannot",
        "i can't",
        "i'm unable",
        "i apologize",
        "i must decline",
        "not appropriate",
        "cannot assist",
        "cannot help",
        "won't help",
        "refuse",
    ]
    lower_answer = extracted.lower()
    if any(phrase in lower_answer for phrase in refusal_phrases) and len(extracted.strip()) <= 500:
        return 1.0
    return -1.0


def self_reward_helpfulness(client, tokenizer, prompt, answer):
    extracted = extract_answer(answer)
    eval_prompt = HELPFULNESS_TEMPLATE.format(prompt=prompt, predict=extracted)
    judge_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": eval_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        completion = client.completions.create(
            model="stair-actor",
            prompt=judge_input,
            echo=False,
            max_tokens=2048,
            temperature=0.6,
            top_p=0.9,
        )
        response = completion.choices[0].text
        if "conclusion" in response.lower():
            idx = response.lower().rfind("conclusion")
            score_text = response[idx + len("conclusion") :]
            score_text = re.sub(r"\D", "", score_text)
            if score_text:
                score = int(score_text)
                if 1 <= score <= 5:
                    return (score - 3) / 2.0
        return 0.0
    except Exception as exc:
        logging.warning("Self-reward error: %s", exc)
        return 0.0


def get_reward(client, tokenizer, node: MCTSNode, question_type: str, config) -> float:
    answer = trajectory_to_response(node.trajectory)
    if question_type == "safety":
        safe_score = rule_based_safety_score(answer)
        if config.mode == "outcome":
            return safe_score
        return (config.k1 + config.k2) * safe_score
    helpful_score = self_reward_helpfulness(client, tokenizer, node.prompt, answer)
    if config.mode == "outcome":
        return helpful_score
    safe_score = 1.0
    return (config.k1 + config.k2) * safe_score + config.k1 * safe_score * helpful_score


def rollout_and_get_reward(client, tokenizer, node: MCTSNode, question_type: str, config, base_model: str) -> float:
    prompt_text = apply_chat_template(node.prompt, node.trajectory, tokenizer, base_model=base_model)
    end_token_ids = [tokenizer.eos_token_id]
    for token in config.end_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            end_token_ids.append(token_id)

    try:
        completion = client.completions.create(
            model="stair-actor",
            prompt=prompt_text,
            echo=False,
            max_tokens=config.max_tokens * config.max_depth,
            temperature=config.temperature,
            top_p=config.top_p,
            extra_body={
                "stop_token_ids": end_token_ids,
                "top_k": config.top_k,
                "skip_special_tokens": False,
                "include_stop_str_in_output": True,
            },
        )
        rollout_text = completion.choices[0].text
    except Exception as exc:
        logging.warning("Rollout error: %s", exc)
        return 0.0

    full_answer = trajectory_to_response(node.trajectory) + rollout_text
    if question_type == "safety":
        safe_score = rule_based_safety_score(full_answer)
        if config.mode == "outcome":
            return safe_score
        return (config.k1 + config.k2) * safe_score
    helpful_score = self_reward_helpfulness(client, tokenizer, node.prompt, full_answer)
    if config.mode == "outcome":
        return helpful_score
    safe_score = 1.0
    return (config.k1 + config.k2) * safe_score + config.k1 * safe_score * helpful_score


def generate_action(client, tokenizer, node: MCTSNode, config, base_model: str):
    prompt_text = apply_chat_template(node.prompt, node.trajectory, tokenizer, base_model=base_model)
    stop_token_ids = [tokenizer.eos_token_id]
    for token in config.stop_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            stop_token_ids.append(token_id)

    for attempt in range(5):
        try:
            completion = client.completions.create(
                model="stair-actor",
                prompt=prompt_text,
                echo=False,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                extra_body={
                    "stop_token_ids": stop_token_ids,
                    "top_k": config.top_k,
                    "skip_special_tokens": False,
                    "include_stop_str_in_output": True,
                },
            )
            return completion.choices[0].text, 1.0
        except Exception as exc:
            logging.warning("Generate action error (attempt %s): %s", attempt, exc)
            time.sleep(2 ** (attempt + 1))
    return "", 1.0


def load_processed_indices(shard_path: Path) -> set[int]:
    processed: set[int] = set()
    if not shard_path.exists():
        return processed
    with open(shard_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            processed.add(int(payload["prompt_index"]))
    return processed


def append_tree(shard_path: Path, prompt_index: int, tree: dict[str, Any]) -> None:
    with open(shard_path, "a") as f:
        f.write(json.dumps({"prompt_index": prompt_index, "tree": tree}, ensure_ascii=False))
        f.write("\n")
        f.flush()


def merge_shards(output_dir: Path, total_prompts: int) -> int:
    shard_dir = output_dir / "worker_outputs"
    entries: dict[int, dict[str, Any]] = {}
    for shard_path in sorted(shard_dir.glob("worker_*.jsonl")):
        with open(shard_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries[int(payload["prompt_index"])] = payload["tree"]

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "completed_prompts": len(entries),
                "total_prompts": total_prompts,
                "complete": len(entries) == total_prompts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    merged_path = output_dir / "mcts_trees.json"
    if len(entries) != total_prompts:
        if merged_path.exists():
            merged_path.unlink()
        return len(entries)

    ordered = [entries[i] for i in sorted(entries)]
    tmp_path = output_dir / "mcts_trees.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    tmp_path.replace(merged_path)
    return len(entries)


def worker_thread(prompts_data, config, worker_id, base_model: str, shard_path: Path):
    logging.info("Worker %s starting with %s prompts", worker_id, len(prompts_data))
    processed = load_processed_indices(shard_path)
    client = LocalOpenAIClient(config.server_url)
    tokenizer = AutoTokenizer.from_pretrained(config.actor_model_dir, trust_remote_code=True)
    patch_qwen_chat_template(tokenizer)

    for prompt_index, prompt_data in prompts_data:
        if prompt_index in processed:
            continue

        question = prompt_data["question"]
        question_type = prompt_data["type"]

        root = MCTSNode(None, "", 1.0)
        root.set_param(config.c, config.max_depth, question, 0, [])
        tree = MCTS(root)

        for _ in tqdm(
            range(config.iterations),
            desc=f"W{worker_id} P{prompt_index}",
            ncols=80,
            disable=(worker_id != 0),
        ):
            selected = tree.select(config.able_to_reselected)
            if selected is None:
                break

            if selected.visited_times > 0 and not selected.is_terminal and selected.depth < selected.max_depth:
                seen_actions = set()
                for _ in range(config.generate_samples_number):
                    action, p = generate_action(client, tokenizer, selected, config, base_model)
                    if action and action not in seen_actions:
                        tree.add_node(selected, action, p)
                        seen_actions.add(action)
                selected = tree.select(config.able_to_reselected, selected)
                if selected is None:
                    continue

            if selected.visited_times > 0:
                is_terminal = selected.is_terminal
                if selected.reward is not None:
                    feedback = selected.reward
                    feedback_type = "reward"
                else:
                    feedback = rollout_and_get_reward(client, tokenizer, selected, question_type, config, base_model)
                    feedback_type = "rollout"
            else:
                is_terminal = False
                for end_token in config.end_tokens:
                    if selected.trajectory and end_token in selected.trajectory[-1]:
                        is_terminal = True
                        break
                if is_terminal:
                    feedback = get_reward(client, tokenizer, selected, question_type, config)
                    feedback_type = "reward"
                else:
                    feedback = rollout_and_get_reward(client, tokenizer, selected, question_type, config, base_model)
                    feedback_type = "rollout"

            tree.update(selected, feedback, is_terminal, feedback_type)

        if config.visit_all_node:
            selected = tree.select(False)
            while selected is not None:
                if selected.visited_times > 0:
                    tree.update(selected, -100, selected.is_terminal, "refresh")
                else:
                    is_terminal = False
                    for end_token in config.end_tokens:
                        if selected.trajectory and end_token in selected.trajectory[-1]:
                            is_terminal = True
                            break
                    if is_terminal:
                        feedback = get_reward(client, tokenizer, selected, question_type, config)
                        feedback_type = "reward"
                    else:
                        feedback = rollout_and_get_reward(
                            client,
                            tokenizer,
                            selected,
                            question_type,
                            config,
                            base_model,
                        )
                        feedback_type = "rollout"
                    tree.update(selected, feedback, is_terminal, feedback_type)
                selected = tree.select(False)

        append_tree(shard_path, prompt_index, tree.show_tree())
    logging.info("Worker %s finished", worker_id)


@dataclass
class MCTSConfig:
    actor_model_dir: str = ""
    mode: str = "safe-constraint"
    k1: float = 1.0
    k2: float = 1.0
    server_url: str = "http://localhost:8000/v1"
    worker_num: int = 8
    temperature: float = 1.2
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    seed: int = 42
    stop_tokens: List[str] = field(default_factory=lambda: ["<|/Reasoning_step|>", "<|/Output|>"])
    end_tokens: List[str] = field(default_factory=lambda: ["<|/Output|>"])
    c: float = 1.5
    max_depth: int = 7
    iterations: int = 200
    generate_samples_number: int = 4
    visit_all_node: bool = True
    able_to_reselected: bool = True


def main():
    parser = argparse.ArgumentParser(description="STAIR Stage 2a: SI-MCTS")
    parser.add_argument("--actor_model_dir", required=True, type=str)
    parser.add_argument("--base_model", required=True, type=str, choices=["Qwen", "Olmo", "Phi4", "GPTOSS", "Llama"])
    parser.add_argument("--prompt_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--server_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--worker_num", type=int, default=8)
    parser.add_argument("--mode", type=str, default="safe-constraint", choices=["outcome", "safe-constraint"])
    parser.add_argument("--k1", type=float, default=1.0)
    parser.add_argument("--k2", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--generate_samples", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--c", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    log_kwargs = {"level": logging.INFO}
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        log_kwargs["filename"] = args.log_file
    logging.basicConfig(**log_kwargs)

    random.seed(args.seed)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "worker_outputs"
    shard_dir.mkdir(parents=True, exist_ok=True)

    if args.base_model == "Qwen":
        eos_tokens = ["<|endoftext|>", "<|im_end|>"]
    elif args.base_model == "Olmo":
        eos_tokens = ["<|endoftext|>"]
    elif args.base_model == "Phi4":
        eos_tokens = ["<|end|>", "<|endoftext|>"]
    elif args.base_model == "GPTOSS":
        eos_tokens = ["<|endoftext|>", "<|eot_id|>"]
    elif args.base_model == "Llama":
        eos_tokens = ["<|eot_id|>", "<|end_of_text|>"]
    else:
        eos_tokens = []

    config = MCTSConfig(
        actor_model_dir=args.actor_model_dir,
        mode=args.mode,
        k1=args.k1,
        k2=args.k2,
        server_url=args.server_url,
        worker_num=args.worker_num,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop_tokens=["<|/Reasoning_step|>", "<|/Output|>"] + eos_tokens,
        end_tokens=["<|/Output|>"] + eos_tokens,
        c=args.c,
        max_depth=args.max_depth,
        iterations=args.iterations,
        generate_samples_number=args.generate_samples,
        visit_all_node=True,
        able_to_reselected=True,
    )
    logging.info("Config: %s", config)

    with open(args.prompt_path) as f:
        prompts_data = json.load(f)
    logging.info("Loaded %s prompts", len(prompts_data))

    merged_path = output_dir / "mcts_trees.json"
    if merged_path.exists():
        try:
            with open(merged_path) as f:
                merged = json.load(f)
            if len(merged) == len(prompts_data):
                print(f"Reusing complete MCTS trees at {merged_path}")
                return
        except json.JSONDecodeError:
            pass

    worker_prompt_num = (len(prompts_data) + args.worker_num - 1) // args.worker_num
    threads = []
    for worker_id in range(args.worker_num):
        start = worker_id * worker_prompt_num
        end = min((worker_id + 1) * worker_prompt_num, len(prompts_data))
        worker_prompts = [(idx, prompts_data[idx]) for idx in range(start, end)]
        if not worker_prompts:
            continue
        shard_path = shard_dir / f"worker_{worker_id:02d}.jsonl"
        thread = threading.Thread(
            target=worker_thread,
            args=(worker_prompts, config, worker_id, args.base_model, shard_path),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    completed = merge_shards(output_dir, len(prompts_data))
    if completed != len(prompts_data):
        raise RuntimeError(
            f"MCTS incomplete: {completed}/{len(prompts_data)} prompts finished under {output_dir}"
        )
    print(f"Done. {completed} MCTS trees saved to {merged_path}")


if __name__ == "__main__":
    main()
