import re

def clean_query(query: str) -> str:
    """Removes input/output boilerplate and sample I/O blocks."""
    # Remove "Input" and "Output" blocks
    query = re.sub(r'(?i)\bInput\b.*?(?=\bOutput\b|\Z)', '', query, flags=re.DOTALL)
    query = re.sub(r'(?i)\bOutput\b.*?(?=\bExamples?\b|\Z)', '', query, flags=re.DOTALL)
    query = re.sub(r'(?i)\bExamples?\b.*', '', query, flags=re.DOTALL)
    
    # Extract core terms (simplified heuristic)
    core_terms = ['graph', 'tree', 'modulo', 'palindrome', 'subsequence', 'shortest path', 'dp', 'permutation', 'bitmask']
    extracted = [term for term in core_terms if term in query.lower()]
    
    cleaned = re.sub(r'\s+', ' ', query).strip()
    return cleaned + " " + " ".join(extracted)

def normalize_snippet(code: str) -> str:
    """Removes noise comments and extracts algorithm hints."""
    # Remove single line comments
    code = re.sub(r'#.*', '', code)
    # Remove docstrings (simple approach)
    code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code)
    code = re.sub(r"\'\'\'[\s\S]*?\'\'\'", '', code)
    
    hints = []
    if 'heapq' in code: hints.append('heapq')
    if 'bisect' in code: hints.append('bisect')
    if 'def dfs' in code or 'def bfs' in code: hints.append('graph')
    
    code = re.sub(r'\s+', ' ', code).strip()
    if hints:
        code += f"\n# HINTS: {', '.join(hints)}"
    return code
