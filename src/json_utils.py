"""Robust JSON extraction from LLM responses."""

import json
import re


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(response: str) -> dict:
    """Parse a JSON object from an LLM response.

    Tolerates:
      - Markdown code fences (``` or ```json)
      - Leading/trailing prose
      - Trailing commas (handled by a second attempt)
    Raises json.JSONDecodeError if no JSON object can be located.
    """
    if not response:
        raise json.JSONDecodeError("empty response", "", 0)
    text = response.strip()

    # 1) direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) strip markdown fences
    m = _CODE_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3) grab the first {...} block
    m = _OBJECT_RE.search(text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # remove trailing commas before } or ]
            cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)
            return json.loads(cleaned)

    raise json.JSONDecodeError("no JSON object found", text, 0)
