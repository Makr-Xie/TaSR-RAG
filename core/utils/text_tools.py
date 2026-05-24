import json
import ast
import re
from typing import Any, Dict, List, Union

def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags from model output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if '<think>' in text:
        json_start = min(
            text.find('{') if text.find('{') != -1 else len(text),
            text.find('[') if text.find('[') != -1 else len(text)
        )
        think_start = text.find('<think>')
        if think_start != -1 and think_start < json_start < len(text):
            text = text[json_start:]
    return text.strip()


def extract_json_str(raw: str) -> str:
    """Extract the first valid JSON object or array string by matching braces."""
    raw = raw.strip()
    l_curly = raw.find('{')
    l_square = raw.find('[')

    if l_curly == -1 and l_square == -1:
        return raw

    if l_curly != -1 and (l_square == -1 or l_curly < l_square):
        start = l_curly
        end_char = '}'
        start_char = '{'
    else:
        start = l_square
        end_char = ']'
        start_char = '['

    count = 0
    in_string = False
    escape = False

    for i in range(start, len(raw)):
        char = raw[i]
        if char == '"' and not escape:
            in_string = not in_string
        if not in_string:
            if char == start_char:
                count += 1
            elif char == end_char:
                count -= 1
                if count == 0:
                    return raw[start : i+1]

        if char == '\\' and not escape:
            escape = True
        else:
            escape = False

    return raw[start:]  # best effort if unclosed


def extract_json_obj(raw: str) -> Any:
    """Extract and parse JSON object/array from string."""
    text = extract_json_str(raw)
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            l = raw.find("{")
            r = raw.rfind("}")
            if l != -1 and r != -1 and l < r:
                try:
                    return json.loads(raw[l:r+1])
                except:
                    pass
            raise ValueError(f"Could not parse JSON from: {raw[:100]}...")


def loads_json_or_py(raw: str) -> Any:
    """Strip thinking tags then parse JSON or Python literal from LLM output."""
    clean = strip_thinking_tags(raw)
    return extract_json_obj(clean)
