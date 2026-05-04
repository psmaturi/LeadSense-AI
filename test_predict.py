import sys
from pathlib import Path

# Add project root to path to ensure imports work correctly
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from inference.hybrid_predictor import predict
except ImportError as e:
    print(f"[ERROR] Could not import prediction engine: {e}")
    print("Ensure you are running this script from the 'Genai Project' root directory.")
    sys.exit(1)

def main():
    print("=" * 60)
    print("  LEADSENSE AI -- MANUAL TEST INTERFACE")
    print("  Type 'exit' to quit.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nEnter lead description: ").strip()

            if user_input.lower() == 'exit':
                print("Exiting test interface. Goodbye!")
                break
            
            if not user_input:
                continue

            # Run prediction
            result = predict(user_input)

            # Print results in a clean format
            print("\nPrediction Result:")
            print(f"  Label      : {result.get('label', 'N/A')}")
            print(f"  Confidence : {result.get('confidence', 0.0):.2f}")
            print(f"  Method     : {result.get('method', 'N/A')}")
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] Prediction failed: {e}")

if __name__ == "__main__":
    main()
