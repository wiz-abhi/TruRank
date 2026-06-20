import json
import time
from src.honeypot_detector import HoneypotDetector

def test_honeypots():
    detector = HoneypotDetector()
    file_path = "data/raw/candidates.jsonl"
    
    print("Running honeypot detector on 100k candidates...")
    start_time = time.time()
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_data = json.loads(line)
            result = detector.detect(raw_data)
            if result.is_honeypot:
                count += 1
                
    print(f"\n--- FINAL RESULTS ---")
    print(f"Total Honeypots Detected: {count}")
    print(f"Time Taken: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    test_honeypots()
