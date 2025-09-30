from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

Grid = List[List[int]]

class Example(BaseModel):
    input: Grid
    output: Grid

class Challenge(BaseModel):
    task_id: str
    train: List[Example]
    test: List[Grid]

class Instructions(BaseModel):
    text: str = Field(..., description="Plain-English instructions describing the transformation.")

class FollowOutput(BaseModel):
    grid: Grid

class Attempt(BaseModel):
    attempt_1: Grid
    attempt_2: Grid

class NLConfig(BaseModel):
    nl_temperature: float = 0.6
    follow_temperature: float = 0.2
    nl_candidates: int = 2
    follow_samples: int = 2
    max_tokens: int = 4096

class EvoConfig(BaseModel):
    # population/evolution
    population_size: int = 12
    generations: int = 4
    elite_k: int = 4
    mutations_per_parent: int = 2

    # per-role models (Ollama model names)
    proposer_model: Optional[str] = None  # defaults to OLLAMA_MODEL if None
    checker_model: Optional[str] = None   # defaults to OLLAMA_MODEL if None

    # entropy schedule
    nl_temperature_start: float = 0.9
    nl_temperature_end: float = 0.5

    # follow/consensus
    follow_temperature: float = 0.2
    follow_samples: int = 3
    follow_consensus_n: int = 2
    max_tokens: int = 4096
    retries: int = 2

class Candidate(BaseModel):
    instructions: str
    fitness: float
