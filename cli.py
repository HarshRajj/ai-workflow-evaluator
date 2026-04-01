import argparse
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
from rich.progress import track
from dotenv import load_dotenv

from parser import TranscriptParser
from evaluator import WorkflowEvaluator

console = Console()

def print_evaluation(result, filename: str):
    console.print(f"\n[bold title]🚀 Workflow Evaluation Report for:[/] [cyan]{filename}[/]")
    console.print("="*60)
    
    # Overall Score
    color = "green" if result.overall_effectiveness_score >= 80 else "yellow" if result.overall_effectiveness_score >= 60 else "red"
    console.print(f"\n[bold]Overall Effectiveness Score:[/] [{color}]{result.overall_effectiveness_score}/100[/]")
    
    # Qualitative Summary
    console.print(Panel(result.qualitative_summary, title="🧠 Qualitative Summary", border_style="blue"))
    
    # Metrics Table
    table = Table(title="📊 Detailed Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=25)
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Reasoning")
    
    for metric in result.metrics:
        score_color = "green" if metric.score >= 80 else "yellow" if metric.score >= 60 else "red"
        table.add_row(
            metric.name,
            f"[{score_color}]{metric.score}/100[/]",
            f"{metric.confidence}%",
            metric.reasoning
        )
    console.print(table)
    
    # Timeline
    console.print("\n[bold]⏱️ Timeline / Phase Breakdown:[/]")
    for phase in result.timeline:
        console.print(f"  [bold yellow]{phase.turn_range}[/] - [cyan]{phase.phase_name}[/]")
        console.print(f"  [dim]{phase.summary}[/]\n")
        
    # Strengths and Improvements
    console.print("[bold green]🌟 Key Strengths:[/]")
    for strength in result.key_strengths:
        console.print(f"  [green]✔ {strength}[/]")
        
    console.print("\n[bold red]📈 Areas for Improvement:[/]")
    for area in result.areas_for_improvement:
        console.print(f"  [red]⚠ {area}[/]")

def main():
    parser = argparse.ArgumentParser(description="AI Workflow Evaluator (CLI)")
    parser.add_argument("path", help="Path to a transcript file or a directory containing markdown transcripts")
    parser.add_argument("--api-key", help="OpenAI API Key (Or use .env file)", default=None)
    
    args = parser.parse_args()
    
    # Load env variables locally 
    load_dotenv()
    
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[bold red]❌ Error: No OPENAI_API_KEY found. Please provide --api-key or place it in a .env file.[/]")
        sys.exit(1)
        
    target_path = Path(args.path)
    if not target_path.exists():
        console.print(f"[bold red]❌ Error: File or directory '{target_path}' not found.[/]")
        sys.exit(1)
        
    evaluator = WorkflowEvaluator(api_key=api_key)
    transcript_parser = TranscriptParser()
    
    files_to_process = []
    if target_path.is_file():
        files_to_process.append(target_path)
    elif target_path.is_dir():
        for ext in ["*.md", "*.txt"]:
            files_to_process.extend(target_path.glob(ext))
            
    if not files_to_process:
        console.print(f"[bold yellow]⚠️ No markdown/text transcripts found in '{target_path}'.[/]")
        sys.exit(0)
        
    console.print(f"[bold cyan]🔍 Found {len(files_to_process)} transcript(s) to evaluate...[/]")
    
    for file in files_to_process:
        try:
            with console.status(f"[bold green]Parsing & Evaluating '{file.name}' via LLM...[/]", spinner="dots"):
                turns = transcript_parser.parse_markdown_file(str(file))
                if not turns:
                    console.print(f"[bold red]❌ Failed to parse any conversation turns from {file.name}.[/]")
                    continue
                
                result = evaluator.evaluate(turns)
                
            print_evaluation(result, file.name)
            
        except Exception as e:
            console.print(f"[bold red]❌ Error evaluating {file.name}: {str(e)}[/]")

if __name__ == "__main__":
    main()
