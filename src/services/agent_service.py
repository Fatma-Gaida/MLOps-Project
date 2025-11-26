from groq import Groq
from src.services.tfidf_services import TFIDFService
from src.services.transformer_service import TransformerService

class AgentService:
    def __init__(self):
        print("🤖 Using Groq Inference API for routing...")

        # Initialize Groq client
        self.client = Groq(api_key="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")

        self.tfidf_service = TFIDFService()
        self.transformer_service = TransformerService()

    def decide_with_groq(self, text: str) -> str:
        """
        Ask Groq model to decide routing.
        """
        prompt = f"""
You are an expert intelligent routing system that decides whether to use TFIDF or TRANSFORMER to classify a text.

Rules:
1. TFIDF is used only for short (<=20 words), simple English texts.
2. TRANSFORMER is used for:
   - Texts longer than 20 words
   - Texts containing technical, nuanced, or complex meaning
   - Texts in any language other than English
3. Always respond with exactly one word: TFIDF or TRANSFORMER. Do NOT include any extra explanation.

Examples:
Text: "Forgot password on my laptop"
Answer: TFIDF

Text: "Le système plante lorsque plusieurs utilisateurs téléchargent de gros fichiers simultanément"
Answer: TRANSFORMER

Text: "Cannot access Outlook account from company laptop"
Answer: TFIDF

Text: "The system crashes when multiple users upload large datasets concurrently"
Answer: TRANSFORMER

Now decide for the following text:
Text: "{text}"
Answer:
"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.7
        )

        result = response.choices[0].message.content.strip().upper()

        if "TRANSFORMER" in result:
            return "transformer"
        return "tfidf"

    def route_text(self, text: str):
        """
        Route text to TF-IDF or Transformer.
        """
        decision = self.decide_with_groq(text)

        if decision == "tfidf":
            prediction = self.tfidf_service.predict(text)
        else:
            prediction = self.transformer_service.generate(text)

        return {
            "chosen_model": decision,
            "prediction": prediction
        }
