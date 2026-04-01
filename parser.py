import re
from pathlib import Path
from typing import List, Dict

class TranscriptParser:
    """Parses markdown transcripts of AI coding sessions into structured turns."""

    def parse_markdown_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Reads a markdown file and extracts conversation turns.
        Returns a list of dicts: [{'role': 'user'|'assistant', 'content': '...'}]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
            
        content = path.read_text(encoding='utf-8')
        
        # We will split by common headings like ### User Input and ### Planner Response
        # We also want to capture system actions like _Edited relevant file_ as assistant content.
        
        # Regex to find all section headers. 
        # Supports standard headings that act as role delineators.
        pattern = re.compile(r'^(#+\s+(?:User Input|Planner Response|Assistant|User))\s*$', re.MULTILINE | re.IGNORECASE)
        
        parts = pattern.split(content)
        
        # parts[0] is everything before the first heading.
        # parts[1] is the first heading '### User Input'
        # parts[2] is the content
        # parts[3] is the next heading, etc.
        
        turns = []
        
        for i in range(1, len(parts), 2):
            heading = parts[i].lower()
            text_content = parts[i+1].strip()
            
            if not text_content:
                continue
                
            role = 'user' if 'user' in heading else 'assistant'
            
            turns.append({
                'role': role,
                'content': text_content
            })
            
        # In case the parsing failed to match standard headings, fallback to looking for 'User:' and 'Assistant:'
        if not turns:
            lines = content.split('\n')
            current_role = None
            current_content = []
            
            for line in lines:
                if line.startswith('User:') or line.startswith('**User:**'):
                    if current_role and current_content:
                        turns.append({'role': current_role, 'content': '\n'.join(current_content).strip()})
                    current_role = 'user'
                    current_content = [line.split(':', 1)[1].strip()] if ':' in line else []
                elif line.startswith('Assistant:') or line.startswith('**Assistant:**') or line.startswith('Planner:'):
                    if current_role and current_content:
                        turns.append({'role': current_role, 'content': '\n'.join(current_content).strip()})
                    current_role = 'assistant'
                    current_content = [line.split(':', 1)[1].strip()] if ':' in line else []
                else:
                    if current_role:
                        current_content.append(line)
                        
            if current_role and current_content:
                turns.append({'role': current_role, 'content': '\n'.join(current_content).strip()})
                
        return turns
