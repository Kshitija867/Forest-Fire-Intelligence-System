import sys
from src.predictor import run_live_inference

def main():
    # Fallback to test coordinates (Algiers region)
    target_lat = 36.7525
    target_lon = 3.0420

    if len(sys.argv) == 3:
        target_lat = float(sys.argv[1])
        target_lon = float(sys.argv[2])

    try:
        weather, indices, prediction, probabilities = run_live_inference(target_lat, target_lon)

        print(f"\nCoordinates: ({target_lat}, {target_lon})")
        print(f"Metrics    : Temp={weather['temp']}°C, RH={weather['rh']}%, Ws={weather['ws']}km/h, Rain={weather['rain']}mm")
        print(f"Indices    : FFMC={indices['FFMC']:.2f}, ISI={indices['ISI']:.2f}")
        print("-" * 50)
        if prediction == 1:
            print(f"⚠️ STATUS: FIRE RISK DETECTED ({probabilities[1]*100:.2f}% Confidence)")
        else:
            print(f" STATUS: SAFE / NO SIGN OF FIRE RISK ({probabilities[0]*100:.2f}% Confidence)")
        print("-" * 50)

    except Exception as e:
        print(f"Execution Failed: {str(e)}")

if __name__ == "__main__":
    main()


