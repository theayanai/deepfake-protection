import os
import json
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    keys_csv = os.getenv("GEMINI_API_KEYS", "")
    keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
    if keys:
        GEMINI_API_KEY = keys[0]

if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY or GEMINI_API_KEYS in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------- CORE FUNCTION ----------------
def detect_synthid(image_path):
    try:
        img = Image.open(image_path)

        prompt = """
        You are an AI forensic analyst.

        Analyze this image and determine:
        - Is it AI-generated or synthetic?
        - Look for artifacts like:
            * unnatural skin texture
            * overly smooth surfaces
            * inconsistent lighting
            * distorted background
            * symmetry artifacts

        Also check for ANY signs of AI watermarking patterns (like SynthID),
        even if not directly visible.

        Return ONLY JSON:
        {
          "ai_generated": true/false,
          "confidence": 0-1,
          "verdict": "YES or NO",
          "reason": "short explanation"
        }
        """

        response = model.generate_content([prompt, img])
        text = response.text.strip()

        # Extract JSON safely
        try:
            result = json.loads(text)
        except:
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                return "ERROR: Could not parse response"

        # Final YES/NO output
        if result.get("ai_generated") or result.get("confidence", 0) > 0.6:
            return f"YES (Likely AI / SynthID present)\nReason: {result.get('reason')}"
        else:
            return f"NO (No strong synthetic signal)\nReason: {result.get('reason')}"

    except Exception as e:
        return f"ERROR: {str(e)}"


# ---------------- RUN ----------------
if __name__ == "__main__":
    path = input("Enter image path: ").strip()

    if not os.path.exists(path):
        print("Invalid path")
    else:
        result = detect_synthid(path)
        print("\nResult:")
        print(result)