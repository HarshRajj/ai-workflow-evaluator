# AI Workflow Evaluator

A lightweight CLI tool that evaluates an engineer's AI-assisted coding transcripts, producing a structured, quantifiable report on workflow quality based on exactly 10 specific engineering metrics:

- **Prompting Quality**
- **Planning**
- **Judgement**
- **Debugging**
- **Tool Usage**
- **Problem Decomposition**
- **Iteration Efficiency**
- **Context Management**
- **Verification & Testing**
- **Autonomy Balance**

The script leverages OpenAI's `gpt-4o-mini` with Pydantic structured outputs for robust parsing.

## Setup

1. **Install [`uv`](https://docs.astral.sh/uv/)** (the fast Python package manager, if you don't have it already).
2. **Set up your API Key:**
   Copy `.env.example` to `.env` and add your OpenAI Key.

   ```bash
   cp .env.example .env
   # Add your key -> OPENAI_API_KEY=sk-...

   ```

3. add the transcripts to the `data` folder

   ```
   data/
   ├── transcript1.md
   ├── transcript2.md
   └── ...
   ```

## Usage

Simply pass a markdown transcript (or a folder of them) to the CLI:

```bash
uv run cli.py data/
```

The script will parse the conversation, evaluate the engineering workflow quality, and print a formatted summary report to your terminal!

**✨ Bonus Feature: Multi-Session Meta-Evaluation**
If you pass a folder containing multiple transcript files, the script will automatically aggregate all evaluations and run a final meta-analysis. This compares the engineer's sessions over time to generate a trajectory report outlining:
- Overall workflow improvement
- Consistent engineering strengths
- Persistent issues that appear in multiple sessions
