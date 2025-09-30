from __future__ import annotations
import json
from statistics import median
from typing import List

from .arc_types import (
    Example, Instructions, FollowOutput, Grid,
    NLConfig, EvoConfig, Candidate
)
from .prompts import INTUITIVE_PROMPT, FOLLOW_PROMPT, REVISION_PROMPT, POOLING_PROMPT
from .llm import chat_text, chat_json
from .logging_basic import info, debug, span, progress, set_ctx
from .utils import grid_equal, grid_similarity

# ---------- baseline ----------

async def propose_instructions(train: List[Example], cfg: NLConfig) -> List[Instructions]:
    with span("nl.propose", candidates=cfg.nl_candidates, temp=cfg.nl_temperature):
        messages = [
            {"role": "system", "content": "You are an expert ARC puzzle solver."},
            {"role": "user", "content": INTUITIVE_PROMPT},
            {"role": "user", "content": f"Training pairs (JSON): {[e.model_dump() for e in train]}"},
            {"role": "user", "content": "Return only the instruction text."},
        ]
        outs: List[Instructions] = []
        for i in range(cfg.nl_candidates):
            txt = await chat_text(messages, temperature=cfg.nl_temperature, max_tokens=cfg.max_tokens)
            outs.append(Instructions(text=txt.strip()))
            progress("nl.propose.progress", current=i + 1, total=cfg.nl_candidates, label="propose")
        info("nl.propose.done", produced=len(outs))
        return outs

async def follow_once(instructions: Instructions, input_grid: Grid, cfg: NLConfig) -> Grid:
    messages = [
        {"role": "system", "content": "You transform grids exactly as instructed."},
        {"role": "user", "content": FOLLOW_PROMPT},
        {"role": "user", "content": f"Instructions:\n{instructions.text}"},
        {"role": "user", "content": f"Input grid (JSON): {input_grid}"},
    ]
    with span("follow.apply"):
        obj = await chat_json(messages, temperature=cfg.follow_temperature, max_tokens=cfg.max_tokens)
        out = FollowOutput.model_validate(obj)
        return out.grid

async def score_instructions_loocv(train: List[Example], instr: Instructions, cfg: NLConfig) -> float:
    with span("score.loocv"):
        exact = 0
        sims: List[float] = []
        for i, held in enumerate(train):
            pred = await follow_once(instr, held.input, cfg)
            e = int(grid_equal(pred, held.output))
            s = grid_similarity(pred, held.output)
            debug("score.example", idx=i, exact=e, similarity=round(s, 4))
            exact += e
            sims.append(s)
            progress("score.progress", current=i + 1, total=len(train), label="score")
        score = 0.8 * (exact / len(train)) + 0.2 * (sum(sims) / len(sims))
        info("score.done", exact=exact, total=len(train), score=round(score, 4), score_percent=round(score * 100, 2))
        return score

async def choose_best_instructions(train: List[Example], cfg: NLConfig) -> Instructions:
    with span("nl.single_best"):
        cands = await propose_instructions(train, cfg)
        scores = []
        for idx, c in enumerate(cands):
            s = await score_instructions_loocv(train, c, cfg)
            scores.append(s)
            progress("nl.single_best.progress", current=idx + 1, total=len(cands), label="pick-best")
        best_idx = max(range(len(cands)), key=lambda i: scores[i])
        best = scores[best_idx]
        info("nl.single_best.done", best=round(best, 4), best_percent=round(best * 100, 2))
        return cands[best_idx]

# ---------- Jeremy-style evo with live % ----------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

async def _propose_population(train: List[Example], evo: EvoConfig) -> List[Instructions]:
    messages = [
        {"role": "system", "content": "You are an expert ARC puzzle solver."},
        {"role": "user", "content": INTUITIVE_PROMPT},
        {"role": "user", "content": f"Training pairs (JSON): {[e.model_dump() for e in train]}"},
        {"role": "user", "content": "Return only the instruction text. Be specific but generalizable."},
    ]
    outs: List[Instructions] = []
    with span("evo.propose", pop=evo.population_size, temp=evo.nl_temperature_start):
        for i in range(evo.population_size):
            txt = await chat_text(
                messages, temperature=evo.nl_temperature_start, max_tokens=evo.max_tokens, model=evo.proposer_model
            )
            outs.append(Instructions(text=txt.strip()))
            progress("evo.propose.progress", current=i + 1, total=evo.population_size, label="evo:propose")
        info("evo.propose.done", produced=len(outs))
    return outs

async def _checker_follow(instructions: Instructions, input_grid: Grid, evo: EvoConfig) -> Grid:
    messages = [
        {"role": "system", "content": "You are a precise grid transformer."},
        {"role": "user", "content": FOLLOW_PROMPT},
        {"role": "user", "content": f"Instructions:\n{instructions.text}"},
        {"role": "user", "content": f"Input grid (JSON): {input_grid}"},
    ]
    with span("evo.follow"):
        obj = await chat_json(
            messages, temperature=evo.follow_temperature, max_tokens=evo.max_tokens, model=evo.checker_model
        )
        out = FollowOutput.model_validate(obj)
        return out.grid

async def _fitness(train: List[Example], instr: Instructions, evo: EvoConfig) -> float:
    exact = 0
    sims: List[float] = []
    with span("evo.fitness"):
        for i, ex in enumerate(train):
            pred = await _checker_follow(instr, ex.input, evo)
            e = int(grid_equal(pred, ex.output))
            s = grid_similarity(pred, ex.output)
            exact += e
            sims.append(s)
            debug("evo.fitness.example", idx=i, exact=e, similarity=round(s, 4))
            progress("evo.fitness.progress", current=i + 1, total=len(train), label="evo:fitness")
        score = 0.8 * (exact / len(train)) + 0.2 * (sum(sims) / len(sims))
        info("evo.fitness.done", exact=exact, total=len(train), score=round(score, 4), score_percent=round(score * 100, 2))
        return score

async def _revise(instr: Instructions, train: List[Example], evo: EvoConfig) -> Instructions:
    diffs = []
    with span("evo.revise"):
        for ex in train:
            try:
                pred = await _checker_follow(instr, ex.input, evo)
                if not grid_equal(pred, ex.output):
                    diffs.append({"input": ex.input, "expected": ex.output, "actual": pred})
            except Exception as e:
                debug("evo.revise.skip", err=str(e))
        if not diffs:
            info("evo.revise.noop")
            return instr
        messages = [
            {"role": "system", "content": "You are a careful ARC instruction editor."},
            {"role": "user", "content": REVISION_PROMPT},
            {"role": "user", "content": f"Current instructions:\n{instr.text}"},
            {"role": "user", "content": f"Training feedback (JSON): {json.dumps(diffs)}"},
        ]
        txt = await chat_text(messages, temperature=0.4, max_tokens=evo.max_tokens, model=evo.proposer_model)
        rev = Instructions(text=txt.strip())
        info("evo.revise.done")
        return rev

async def _mutate(instr: Instructions, train: List[Example], evo: EvoConfig, temperature: float) -> Instructions:
    with span("evo.mutate", temp=temperature):
        messages = [
            {"role": "system", "content": "You are a concise ARC instruction mutator."},
            {"role": "user", "content": "Produce a slightly different version to fix edge cases without losing generality. Return only the instruction text."},
            {"role": "user", "content": f"Original instructions:\n{instr.text}"},
            {"role": "user", "content": f"Training pairs (JSON): {[e.model_dump() for e in train]}"},
        ]
        txt = await chat_text(messages, temperature=temperature, max_tokens=evo.max_tokens, model=evo.proposer_model)
        return Instructions(text=txt.strip())

async def _pool(instructions: List[Instructions], evo: EvoConfig) -> Instructions:
    with span("evo.pool", n=len(instructions)):
        msgs = [
            {"role": "system", "content": "You synthesize concise, correct ARC instructions."},
            {"role": "user", "content": POOLING_PROMPT},
            {"role": "user", "content": "Candidates:\n" + "\n\n---\n\n".join(i.text for i in instructions)},
        ]
        txt = await chat_text(msgs, temperature=0.4, max_tokens=evo.max_tokens, model=evo.proposer_model)
        pooled = Instructions(text=txt.strip())
        info("evo.pool.done")
        return pooled

async def evolve_instructions(train: List[Example], evo: EvoConfig) -> Instructions:
    with span("evo.init", pop=evo.population_size, gens=evo.generations):
        pop = await _propose_population(train, evo)
        scored: List[Candidate] = []
        with span("evo.score_g0"):
            for i, ins in enumerate(pop):
                f = await _fitness(train, ins, evo)
                scored.append(Candidate(instructions=ins.text, fitness=f))
                progress("evo.g0.progress", current=i + 1, total=len(pop), label="evo:g0-score")
        scored.sort(key=lambda c: c.fitness, reverse=True)
        fits = [c.fitness for c in scored]
        best = scored[0].fitness
        info("evo.g0", best=round(best, 4), best_percent=round(best * 100, 2))
        info("evo.g0.stats",
             best=round(best, 4),
             best_percent=round(best * 100, 2),
             mean=round(sum(fits)/len(fits), 4),
             mean_percent=round((sum(fits)/len(fits))*100, 2),
             median=round(median(fits), 4),
             median_percent=round(median(fits)*100, 2),
             worst=round(fits[-1], 4),
             worst_percent=round(fits[-1]*100, 2))
        debug("evo.g0.best_instructions", text=scored[0].instructions[:200])

    for g in range(1, evo.generations + 1):
        set_ctx(gen=g)
        temp = _lerp(evo.nl_temperature_start, evo.nl_temperature_end, g / max(1, evo.generations))
        with span("evo.gen", gen=g, temp=temp):
            elite = scored[:evo.elite_k]
            pooled = await _pool([Instructions(text=c.instructions) for c in elite], evo)

            offsprings: List[Instructions] = [pooled]
            with span("evo.revise_batch", n=len(elite)):
                for idx, parent in enumerate(elite):
                    base = Instructions(text=parent.instructions)
                    offsprings.append(await _revise(base, train, evo))
                    progress("evo.revise_batch.progress", current=idx + 1, total=len(elite), label="evo:revise")

            with span("evo.mutate_batch", n=evo.elite_k * evo.mutations_per_parent, temp=temp):
                total = evo.elite_k * evo.mutations_per_parent
                done = 0
                for parent in elite:
                    base = Instructions(text=parent.instructions)
                    for _ in range(evo.mutations_per_parent):
                        offsprings.append(await _mutate(base, train, evo, temperature=temp))
                        done += 1
                        progress("evo.mutate_batch.progress", current=done, total=total, label="evo:mutate")

            with span("evo.rescore"):
                next_gen: List[Candidate] = elite[:]  # keep elite
                for i, ins in enumerate(offsprings):
                    f = await _fitness(train, ins, evo)
                    next_gen.append(Candidate(instructions=ins.text, fitness=f))
                    progress("evo.rescore.progress", current=i + 1, total=len(offsprings), label="evo:rescore")
                next_gen.sort(key=lambda c: c.fitness, reverse=True)
                scored = next_gen[:evo.population_size]
                fits = [c.fitness for c in scored]
                best = scored[0].fitness
                info("evo.gen", gen=g, best=round(best, 4), best_percent=round(best * 100, 2))
                info("evo.gen.stats",
                     gen=g,
                     best=round(best, 4),
                     best_percent=round(best * 100, 2),
                     mean=round(sum(fits)/len(fits), 4),
                     mean_percent=round((sum(fits)/len(fits))*100, 2),
                     median=round(median(fits), 4),
                     median_percent=round(median(fits)*100, 2),
                     worst=round(fits[-1], 4),
                     worst_percent=round(fits[-1]*100, 2),
                     temp=temp)
                debug("evo.gen.best_instructions", gen=g, text=scored[0].instructions[:200])

    return Instructions(text=scored[0].instructions)

async def _consensus_follow(ins: Instructions, input_grid: Grid, evo: EvoConfig) -> Grid:
    seen: dict[str, tuple[Grid, int]] = {}
    trials = max(2, evo.follow_samples * 2)
    with span("test.consensus", trials=trials, consensus=evo.follow_consensus_n):
        for t in range(trials):
            g = await _checker_follow(ins, input_grid, evo)
            key = json.dumps(g)
            grid, cnt = seen.get(key, (g, 0))
            cnt += 1
            seen[key] = (grid, cnt)
            debug("test.consensus.tick", t=t + 1, seen=len(seen), count=cnt)
            progress("test.consensus.progress", current=t + 1, total=trials, label="consensus")
            if cnt >= max(1, evo.follow_consensus_n):
                info("test.consensus.lock", t=t + 1, count=cnt, percent=round(100.0 * (t + 1) / trials, 2))
                return grid
        best = max(seen.values(), key=lambda v: v[1])[0] if seen else [[0 for _ in input_grid[0]] for __ in input_grid]
        info("test.consensus.fallback", unique=len(seen), percent=100.0)
        return best

async def predict_test_grids(
    train: List[Example],
    test_inputs: List[Grid],
    cfg: NLConfig,
    evo: EvoConfig | None = None
) -> tuple[Instructions, List[List[Grid]]]:
    if evo is None:
        best = await choose_best_instructions(train, cfg)
        attempts_per_test: List[List[Grid]] = []
        for i, g in enumerate(test_inputs):
            with span("test.simple", idx=i):
                uniq: List[Grid] = []
                total = cfg.follow_samples * 2
                for t in range(total):
                    try:
                        out = await follow_once(best, g, cfg)
                        if not uniq or out != uniq[-1]:
                            uniq.append(out)
                        progress("test.simple.progress", current=t + 1, total=total, label="simple-follow")
                        if len(uniq) >= 2:
                            break
                    except Exception as e:
                        debug("test.simple.error", err=str(e))
                if not uniq:
                    uniq = [[[0 for _ in g[0]] for __ in g]]
                if len(uniq) == 1:
                    uniq.append(uniq[0])
                attempts_per_test.append(uniq[:2])
        return best, attempts_per_test

    best = await evolve_instructions(train, evo)
    attempts_per_test: List[List[Grid]] = []
    for i, g in enumerate(test_inputs):
        with span("test.predict", idx=i):
            a = await _consensus_follow(best, g, evo)
            b_ins = await _mutate(best, train, evo, temperature=max(evo.nl_temperature_end, 0.7))
            b = await _checker_follow(b_ins, g, evo)
            attempts_per_test.append([a, b])
            info("test.predict.done", idx=i, percent=round(100.0 * (i + 1) / max(1, len(test_inputs)), 2))
    return best, attempts_per_test
