import re

def is_valid_input(text: str) -> bool:
    # 1. Reject empty or whitespace input
    text = text.strip()
    if not text:
        return False
    
    # 2. Reject very short input (<5 characters)
    if len(text) < 5:
        return False
    
    # 3. Reject inputs with only numbers (including negative/decimals)
    if re.fullmatch(r"^-?\d+(\.\d+)?$", text):
        return False
    
    # 4. Reject inputs with only special characters
    if re.fullmatch(r"[^a-zA-Z0-9\s]+", text):
        return False
    
    # 5. Reject inputs with less than 2 meaningful words
    words = [w for w in text.split() if re.search(r"[a-zA-Z0-9]", w)]
    if len(words) < 2:
        return False
        
    return True

# Test cases
test_inputs = [
    ("123456", False),
    ("@@@@@@", False),
    ("abc", False),
    ("-123", False),
    ("-45.6", False),
    ("    ", False),
    ("hello", False),
    ("hello world", True),
    ("I am interested", True),
    ("123 help", True),
    ("!!! hello !!! world !!!", True),
    ("1234.5678", False)
]

for inp, expected in test_inputs:
    result = is_valid_input(inp)
    status = "PASS" if result == expected else "FAIL"
    print(f"[{status}] Input: '{inp}' -> Result: {result} (Expected: {expected})")
