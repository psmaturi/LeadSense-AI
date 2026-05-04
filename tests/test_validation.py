import requests

BASE_URL = "http://127.0.0.1:8000"

test_cases = [
    {"text": "123456", "desc": "Numbers only"},
    {"text": "@@@@@@", "desc": "Special characters only"},
    {"text": "abc", "desc": "Too short"},
    {"text": "-123", "desc": "Negative number"},
    {"text": "-45.6", "desc": "Negative decimal"},
    {"text": "    ", "desc": "Whitespace"},
    {"text": "hello", "desc": "Single word"},
    {"text": "hello world", "desc": "Valid input (2 words)"},
    {"text": "I am interested in buying your product", "desc": "Valid input (Sentences)"},
]

def run_tests():
    print(f"{'Description':<25} | {'Input':<15} | {'Status':<10} | {'Label':<10}")
    print("-" * 70)
    for case in test_cases:
        try:
            response = requests.post(f"{BASE_URL}/api/classify", json={"text": case["text"]})
            status = response.status_code
            data = response.json()
            label = data.get("label")
            print(f"{case['desc']:<25} | {case['text']:<15} | {status:<10} | {label:<10}")
        except Exception as e:
            print(f"Error testing {case['text']}: {e}")

if __name__ == "__main__":
    # Note: Ensure the API is running before executing this
    run_tests()
