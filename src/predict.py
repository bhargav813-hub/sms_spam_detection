"""
SMS Spam Detection System - Prediction Module
Loads the trained model and TF-IDF vectorizer, performs inference on new SMS inputs,
and returns predictions along with confidence probabilities.
"""

import os
import sys
import joblib
from typing import Dict, Any

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_text


class SpamPredictor:
    """
    Inference handler for SMS spam classification.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.classifier_path = os.path.join(models_dir, "spam_classifier.pkl")
        self.vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        self.metrics_path = os.path.join(models_dir, "model_metrics.pkl")
        
        self.classifier = None
        self.vectorizer = None
        self.metrics = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model, vectorizer, and evaluation metrics."""
        if not os.path.exists(self.classifier_path) or not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(
                f"Model artifacts not found in '{self.models_dir}'. "
                "Please run 'python src/train_model.py' to train and save models first."
            )

        self.classifier = joblib.load(self.classifier_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        
        if os.path.exists(self.metrics_path):
            try:
                self.metrics = joblib.load(self.metrics_path)
            except Exception:
                self.metrics = None

    def predict(self, sms_text: str) -> Dict[str, Any]:
        """
        Predicts whether an SMS message is Ham (Not Spam) or Spam.
        
        Parameters:
            sms_text (str): Raw SMS text message.
            
        Returns:
            Dict[str, Any]: Prediction result containing:
                - label: 'spam' or 'ham'
                - is_spam: bool
                - confidence: float (percentage 0-100)
                - proba_ham: float (0.0 to 1.0)
                - proba_spam: float (0.0 to 1.0)
                - cleaned_text: str
                - raw_text: str
        """
        # Validate input
        if not sms_text or not isinstance(sms_text, str) or not sms_text.strip():
            return {
                "error": "Input SMS message cannot be empty. Please provide a message to classify.",
                "is_valid": False
            }

        cleaned = preprocess_text(sms_text)
        
        if not cleaned.strip():
            # If all tokens were stripped (e.g. only punctuation or special symbols)
            return {
                "error": "The message does not contain any valid alphanumeric words after preprocessing.",
                "is_valid": False,
                "cleaned_text": cleaned,
                "raw_text": sms_text
            }

        # Vectorize using fitted TF-IDF
        features = self.vectorizer.transform([cleaned])

        # Predict class
        prediction = self.classifier.predict(features)[0]
        is_spam = bool(prediction == 1)
        label = "spam" if is_spam else "ham"

        # Probability calculation
        proba_ham = 0.5
        proba_spam = 0.5
        confidence = 50.0

        if hasattr(self.classifier, "predict_proba"):
            probabilities = self.classifier.predict_proba(features)[0]
            # Probabilities ordered by classes [0, 1] -> [ham, spam]
            proba_ham = float(probabilities[0])
            proba_spam = float(probabilities[1])
            confidence = float(proba_spam * 100 if is_spam else proba_ham * 100)
        elif hasattr(self.classifier, "decision_function"):
            decision = self.classifier.decision_function(features)[0]
            # Convert decision score to sigmoid probability
            proba_spam = float(1 / (1 + (2.718281828459045 ** (-decision))))
            proba_ham = float(1 - proba_spam)
            confidence = float(proba_spam * 100 if is_spam else proba_ham * 100)

        return {
            "is_valid": True,
            "label": label,
            "is_spam": is_spam,
            "confidence": round(confidence, 2),
            "proba_ham": round(proba_ham, 4),
            "proba_spam": round(proba_spam, 4),
            "cleaned_text": cleaned,
            "raw_text": sms_text
        }


# Global helper instance for quick usage
_default_predictor = None

def get_predictor() -> SpamPredictor:
    """Returns a cached instance of SpamPredictor."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = SpamPredictor()
    return _default_predictor


def predict_sms(message: str) -> Dict[str, Any]:
    """
    Convenience function to classify an SMS message.
    """
    predictor = get_predictor()
    return predictor.predict(message)


if __name__ == "__main__":
    # Ensure stdout handles UTF-8 on Windows consoles
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    test_samples = [
        "Hey bro, are we meeting at the college library at 4pm for group study?",
        "URGENT! You have won £1,000 cash prize or a £2,000 holiday! Claim now. Call 09050000327 to claim.",
        "Can you pick up some milk on your way home?",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate).",
    ]

    print("--- Running Test SMS Predictions ---\n")
    for sample in test_samples:
        res = predict_sms(sample)
        tag = "[SPAM]" if res["is_spam"] else "[HAM]"
        print(f"Message: \"{sample}\"")
        print(f"Cleaned: \"{res['cleaned_text']}\"")
        print(f"Result:  {tag} (Confidence: {res['confidence']}% | P(Ham): {res['proba_ham']} | P(Spam): {res['proba_spam']})\n")
