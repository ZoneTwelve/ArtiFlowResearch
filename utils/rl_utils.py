# utils/rl_utils.py

import re

def compute_reward(generated_text, reference_text, pattern=r"<title>(.*?)</title>"):
    """
    Returns 1.0 if `generated_text` fully matches <title>...</title> pattern,
    else -10.0.
    """
    # Construct the exact pattern with the reference text
    exact_pattern = rf"<title>{re.escape(reference_text)}</title>"
    match = re.fullmatch(exact_pattern, generated_text.strip())
    return 1.0 if match else -10.0
