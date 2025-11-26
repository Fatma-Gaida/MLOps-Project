class TransformerService:
    def __init__(self):
        print("🚀 Transformer service initialized (mock or real).")

    def generate(self, text: str):
        """Generate a prediction/response using the transformer model."""
        return {"prediction": f"Predicted by Transformer for input: {text}"}
