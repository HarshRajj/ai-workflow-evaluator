import os
from typing import List, Optional
from openai import OpenAI, AuthenticationError, RateLimitError
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class MetricScore(BaseModel):
    name: str
    score: int = Field(..., ge=0, le=100)
    confidence: int = Field(..., ge=0, le=100)
    reasoning: str


class TimelinePhase(BaseModel):
    phase_name: str
    turn_range: str
    summary: str


class WorkflowEvaluation(BaseModel):
    overall_effectiveness_score: int = Field(..., ge=0, le=100)
    metrics: List[MetricScore]
    timeline: List[TimelinePhase]
    qualitative_summary: str
    key_strengths: List[str]
    areas_for_improvement: List[str]


class SessionComparison(BaseModel):
    improvement_trajectory: str
    consistent_strengths: List[str]
    persistent_issues: List[str]
    meta_summary: str


# ---------------------------------------------------------------------------
# Turn type
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

METRICS = [
    "Prompting Quality",
    "Planning",
    "Judgement",
    "Debugging",
    "Tool Usage",
    "Problem Decomposition",
    "Iteration Efficiency",
    "Context Management",
    "Verification & Testing",
    "Autonomy Balance",
]

SYSTEM_PROMPT = """\
You are a senior engineering productivity analyst specialising in AI-assisted development workflows.
Your evaluations are grounded, specific, and evidence-based — you always cite concrete examples
from the transcript to justify every score.

Scoring philosophy:
- 90–100  : Exceptional. Rare. Reserve for genuinely outstanding behaviour.
- 70–89   : Good. Competent use with minor gaps.
- 50–69   : Average. Works, but clear room for improvement.
- 30–49   : Below average. Noticeable inefficiencies or poor habits.
- 0–29    : Poor. Significant issues that hurt productivity.

Be strict. Do not default to safe mid-range scores. If you cannot find evidence for a
metric, lower both the score AND the confidence.
"""

USER_PROMPT_TEMPLATE = """\
Analyse the following AI-assisted coding session transcript and evaluate the engineer's workflow.

<transcript>
{transcript}
</transcript>

Evaluate the engineer on EXACTLY these {n} metrics (use the exact names):
{metric_list}

Metric definitions:
- Prompting Quality       : Clarity, specificity, and context richness of the engineer's prompts.
- Planning                : Evidence of thinking ahead — structuring problems before asking.
- Judgement               : Critical evaluation of AI output; catching errors; knowing when NOT to follow the AI.
- Debugging               : Systematic diagnosis of failures; quality of bug reports given to the AI.
- Tool Usage              : Effective use of AI features (follow-ups, regeneration, context pinning, etc.).
- Problem Decomposition   : Breaking complex tasks into small, well-scoped sub-tasks.
- Iteration Efficiency    : How quickly the engineer converges on a correct solution; avoids pointless loops.
- Context Management      : Providing relevant background, pasting correct code, avoiding context loss.
- Verification & Testing  : Validating AI suggestions before accepting; running tests; checking edge cases.
- Autonomy Balance        : Healthy balance between leaning on AI and applying own expertise.

For each metric provide:
  - score       (0–100, strict)
  - confidence  (0–100, how certain you are based on transcript evidence)
  - reasoning   (1–3 sentences citing specific turns or patterns)

Also produce:
  - overall_effectiveness_score : weighted holistic score (0–100)
  - timeline                    : ordered list of session phases (e.g. Planning → Implementation → Debugging)
  - qualitative_summary         : 3–5 sentence narrative answering "How effectively is this engineer using AI?"
  - key_strengths               : 2–4 specific, evidence-backed strengths
  - areas_for_improvement       : 2–4 specific, actionable improvement points
"""

META_EVAL_PROMPT_TEMPLATE = """\
Analyze the following sequential workflow evaluations for the same engineer across multiple coding sessions.

<evaluations>
{evaluations_json}
</evaluations>

Your goal is to compare these sessions and identify the trajectory of the engineer's workflow.
Identify what habits improved, what strengths remained consistent, and what areas for improvement persisted.

Provide:
  - improvement_trajectory: A brief narrative (2-3 sentences) on how their workflow evolved.
  - consistent_strengths: 2-3 specific strengths displayed across the sessions.
  - persistent_issues: 2-3 specific weaknesses that appeared in multiple sessions.
  - meta_summary: A 1-2 sentence final verdict on their overall AI interaction skills.
"""


class WorkflowEvaluator:
    MAX_TRANSCRIPT_CHARS = 80_000  # ~20 k tokens — safe ceiling for gpt-4o context

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
    ):
        """
        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model:   OpenAI model to use. Defaults to gpt-4o for richer reasoning.
                     Use "gpt-4o-mini" for faster / cheaper evaluations.
        """
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in your environment "
                "or pass api_key= to WorkflowEvaluator()."
            )
        self.client = OpenAI(api_key=key)
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, turns: List[dict]) -> WorkflowEvaluation:
        """
        Evaluate a coding session transcript.

        Args:
            turns: List of dicts with keys 'role' ('user' | 'assistant') and 'content'.

        Returns:
            WorkflowEvaluation Pydantic object with scores, timeline, and narrative.
        """
        validated = self._validate_turns(turns)
        transcript = self._format_transcript(validated)
        transcript = self._truncate(transcript)
        return self._call_api(transcript)

    def compare_sessions(self, session_results: List[dict]) -> SessionComparison:
        """
        Compare multiple session evaluations.
        Args:
            session_results: list of dicts describing the sessions
        """
        import json
        evaluations_str = json.dumps(session_results, indent=2)
        user_prompt = META_EVAL_PROMPT_TEMPLATE.format(evaluations_json=evaluations_str)
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=SessionComparison,
                temperature=0.2,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            raise RuntimeError(f"OpenAI API error during comparison: {e}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_turns(self, turns: List[dict]) -> List[Turn]:
        validated = []
        for i, t in enumerate(turns):
            role = t.get("role", "")
            content = t.get("content", "")
            if role not in ("user", "assistant"):
                raise ValueError(
                    f"Turn {i} has invalid role '{role}'. Must be 'user' or 'assistant'."
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Turn {i} has empty or non-string content.")
            validated.append(Turn(role=role, content=content.strip()))
        if not validated:
            raise ValueError("Transcript must contain at least one turn.")
        return validated

    def _format_transcript(self, turns: List[Turn]) -> str:
        lines = []
        for idx, turn in enumerate(turns, 1):
            label = "Engineer" if turn.role == "user" else "AI Assistant"
            lines.append(f"[Turn {idx}] {label}:\n{turn.content}\n")
        return "\n".join(lines)

    def _truncate(self, transcript: str) -> str:
        if len(transcript) > self.MAX_TRANSCRIPT_CHARS:
            transcript = (
                transcript[: self.MAX_TRANSCRIPT_CHARS]
                + "\n\n[Transcript truncated due to length]"
            )
        return transcript

    def _build_user_prompt(self, transcript: str) -> str:
        metric_list = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(METRICS))
        return USER_PROMPT_TEMPLATE.format(
            transcript=transcript,
            n=len(METRICS),
            metric_list=metric_list,
        )

    def _call_api(self, transcript: str) -> WorkflowEvaluation:
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": self._build_user_prompt(transcript)},
                ],
                response_format=WorkflowEvaluation,
                temperature=0.2,  # Low temp → more consistent, less hallucinated scores
            )
        except AuthenticationError:
            raise ValueError(
                "Invalid OpenAI API key. Check your OPENAI_API_KEY."
            )
        except RateLimitError:
            raise RuntimeError(
                "OpenAI rate limit reached. Wait a moment and try again."
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

        result = response.choices[0].message.parsed

        # Sanity-check: ensure all expected metrics are present
        returned_names = {m.name for m in result.metrics}
        missing = [m for m in METRICS if m not in returned_names]
        if missing:
            raise RuntimeError(
                f"Model omitted expected metrics: {missing}. "
                "Try again or switch to a larger model."
            )

        return result