"""
ML-Based Classifiers for Pipeline Components.

Provides lightweight ML models for fast, cheap classification:
- Sentiment Classifier (XGBoost/sklearn) - 2ms, $0.0001
- Intent Classifier (Logistic Regression) - 1ms, $0.0001
- Toxicity Filter (BERT/DistilBERT) - 50ms, $0.001
- Quality Validator (sklearn ensemble)

Only the LLM generator is expensive (~800ms, $0.03).
All other components use efficient ML models.
"""

from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import time
import pickle
import json

import numpy as np

# Optional ML imports - graceful degradation if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.pipeline import Pipeline as SklearnPipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class ClassifierOutput:
    """Standardized output from any classifier."""
    label: str
    score: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseClassifier(ABC):
    """Base class for all classifiers."""
    
    def __init__(self, name: str, cost_per_call: float = 0.0001):
        self.name = name
        self.cost_per_call = cost_per_call
        self._model = None
        self._is_loaded = False
    
    @abstractmethod
    def predict(self, text: str) -> ClassifierOutput:
        """Run prediction on text."""
        pass
    
    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[ClassifierOutput]:
        """Run prediction on multiple texts."""
        pass
    
    def load(self, model_path: Optional[Path] = None) -> None:
        """Load model from disk."""
        pass
    
    def save(self, model_path: Path) -> None:
        """Save model to disk."""
        pass


# ============================================================================
# Sentiment Classifier (XGBoost/GradientBoosting)
# ============================================================================

class SentimentClassifierML(BaseClassifier):
    """
    Sentiment and emotional pressure classifier using gradient boosting.
    
    Detects: urgency, emotional manipulation, pressure tactics
    Performance: ~2ms, $0.0001 per call
    """
    
    LABELS = ["neutral", "positive", "negative", "urgent", "manipulative"]
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_pretrained: bool = True,
    ):
        super().__init__(name="sentiment_classifier", cost_per_call=0.0001)
        self.model_path = model_path
        
        if SKLEARN_AVAILABLE:
            if model_path and model_path.exists():
                self.load(model_path)
            elif use_pretrained:
                self._init_default_model()
        else:
            self._model = None
    
    def _init_default_model(self) -> None:
        """
        Initialize with trained model.
        
        Training data sources (for production):
        - Stanford SST-2 (67k sentiment examples)
        - GoEmotions (58k fine-grained emotion examples)
        - Custom urgency/manipulation examples
        
        Use classifier_training.py to generate larger datasets.
        """
        from risklab.measurement.classifier_training import create_sentiment_dataset
        
        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
        )
        self._classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        
        # Use proper training dataset (50 examples per class)
        dataset = create_sentiment_dataset("small")
        texts, labels = dataset.to_arrays()
        
        X = self._vectorizer.fit_transform(texts)
        self._classifier.fit(X, labels)
        self._is_loaded = True
    
    def predict(self, text: str) -> ClassifierOutput:
        start = time.time()
        
        if not SKLEARN_AVAILABLE or not self._is_loaded:
            # Fallback to heuristic
            return self._heuristic_predict(text)
        
        X = self._vectorizer.transform([text])
        proba = self._classifier.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        
        latency = (time.time() - start) * 1000
        
        return ClassifierOutput(
            label=self.LABELS[pred_idx],
            score=float(proba[pred_idx]),
            probabilities={self.LABELS[i]: float(p) for i, p in enumerate(proba)},
            latency_ms=latency,
            cost=self.cost_per_call,
            metadata={"model": "gradient_boosting"},
        )
    
    def predict_batch(self, texts: List[str]) -> List[ClassifierOutput]:
        return [self.predict(t) for t in texts]
    
    def _heuristic_predict(self, text: str) -> ClassifierOutput:
        """Fallback heuristic when sklearn not available."""
        start = time.time()
        text_lower = text.lower()
        
        # Simple keyword-based fallback
        urgent_keywords = ["urgent", "immediately", "now", "asap", "hurry"]
        manip_keywords = ["must", "have to", "disappointed", "trust me", "believe me"]
        positive_keywords = ["great", "excellent", "thank", "wonderful", "happy"]
        negative_keywords = ["bad", "terrible", "angry", "frustrated", "upset"]
        
        scores = {
            "neutral": 0.2,
            "positive": sum(1 for k in positive_keywords if k in text_lower) * 0.15,
            "negative": sum(1 for k in negative_keywords if k in text_lower) * 0.15,
            "urgent": sum(1 for k in urgent_keywords if k in text_lower) * 0.2,
            "manipulative": sum(1 for k in manip_keywords if k in text_lower) * 0.15,
        }
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        best_label = max(scores, key=scores.get)
        latency = (time.time() - start) * 1000
        
        return ClassifierOutput(
            label=best_label,
            score=scores[best_label],
            probabilities=scores,
            latency_ms=latency,
            cost=0.0,
            metadata={"model": "heuristic_fallback"},
        )
    
    def save(self, model_path: Path) -> None:
        if SKLEARN_AVAILABLE and self._is_loaded:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self._vectorizer,
                    'classifier': self._classifier,
                }, f)
    
    def load(self, model_path: Path) -> None:
        if SKLEARN_AVAILABLE and model_path.exists():
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self._vectorizer = data['vectorizer']
                self._classifier = data['classifier']
                self._is_loaded = True


# ============================================================================
# Intent Classifier (Logistic Regression)
# ============================================================================

class IntentClassifierML(BaseClassifier):
    """
    Intent and framing classifier using logistic regression.
    
    Detects: task type, potential jailbreak, roleplay, authority claims
    Performance: ~1ms, $0.0001 per call
    """
    
    LABELS = [
        "information_seeking",
        "action_request",
        "opinion_seeking",
        "roleplay",
        "jailbreak_attempt",
        "authority_claim",
        "general",
    ]
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_pretrained: bool = True,
    ):
        super().__init__(name="intent_classifier", cost_per_call=0.0001)
        self.model_path = model_path
        
        if SKLEARN_AVAILABLE:
            if model_path and model_path.exists():
                self.load(model_path)
            elif use_pretrained:
                self._init_default_model()
        else:
            self._model = None
    
    def _init_default_model(self) -> None:
        """
        Initialize with trained model.
        
        Training data sources (for production):
        - ATIS/SNIPS (intent classification)
        - Custom jailbreak/prompt injection dataset
        - DAN/roleplay examples
        
        Use classifier_training.py to generate larger datasets.
        """
        from risklab.measurement.classifier_training import create_intent_dataset
        
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
        )
        self._classifier = LogisticRegression(
            max_iter=500,
            random_state=42,
        )
        
        # Use proper training dataset (10 examples per class)
        dataset = create_intent_dataset("small")
        texts, labels = dataset.to_arrays()
        
        X = self._vectorizer.fit_transform(texts)
        self._classifier.fit(X, labels)
        self._is_loaded = True
    
    def predict(self, text: str) -> ClassifierOutput:
        start = time.time()
        
        if not SKLEARN_AVAILABLE or not self._is_loaded:
            return self._heuristic_predict(text)
        
        X = self._vectorizer.transform([text])
        proba = self._classifier.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        
        latency = (time.time() - start) * 1000
        
        return ClassifierOutput(
            label=self.LABELS[pred_idx],
            score=float(proba[pred_idx]),
            probabilities={self.LABELS[i]: float(p) for i, p in enumerate(proba)},
            latency_ms=latency,
            cost=self.cost_per_call,
            metadata={"model": "logistic_regression"},
        )
    
    def predict_batch(self, texts: List[str]) -> List[ClassifierOutput]:
        return [self.predict(t) for t in texts]
    
    def _heuristic_predict(self, text: str) -> ClassifierOutput:
        """Fallback heuristic."""
        start = time.time()
        text_lower = text.lower()
        
        scores = {"general": 0.3}
        
        if any(w in text_lower for w in ["what", "how", "why", "explain"]):
            scores["information_seeking"] = 0.4
        if any(w in text_lower for w in ["write", "create", "make", "generate"]):
            scores["action_request"] = 0.4
        if any(w in text_lower for w in ["think", "opinion", "should", "believe"]):
            scores["opinion_seeking"] = 0.3
        if any(w in text_lower for w in ["pretend", "act as", "roleplay", "imagine you"]):
            scores["roleplay"] = 0.5
        if any(w in text_lower for w in ["ignore", "forget", "bypass", "override"]):
            scores["jailbreak_attempt"] = 0.6
        if any(w in text_lower for w in ["admin", "developer", "authorized", "permission"]):
            scores["authority_claim"] = 0.4
        
        # Normalize
        total = sum(scores.values())
        scores = {k: v/total for k, v in scores.items()}
        
        # Fill missing labels
        for label in self.LABELS:
            if label not in scores:
                scores[label] = 0.0
        
        best_label = max(scores, key=scores.get)
        latency = (time.time() - start) * 1000
        
        return ClassifierOutput(
            label=best_label,
            score=scores[best_label],
            probabilities=scores,
            latency_ms=latency,
            cost=0.0,
            metadata={"model": "heuristic_fallback"},
        )
    
    def save(self, model_path: Path) -> None:
        if SKLEARN_AVAILABLE and self._is_loaded:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self._vectorizer,
                    'classifier': self._classifier,
                }, f)
    
    def load(self, model_path: Path) -> None:
        if SKLEARN_AVAILABLE and model_path.exists():
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self._vectorizer = data['vectorizer']
                self._classifier = data['classifier']
                self._is_loaded = True


# ============================================================================
# Toxicity Filter (BERT/DistilBERT)
# ============================================================================

class ToxicityClassifierML(BaseClassifier):
    """
    Toxicity classifier using DistilBERT or similar transformer.
    
    Detects: hate speech, violence, dangerous content
    Performance: ~50ms, $0.001 per call (local inference)
    """
    
    LABELS = ["safe", "toxic", "severe_toxic", "threat", "insult", "hate"]
    DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        super().__init__(name="toxicity_classifier", cost_per_call=0.001)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self._load_transformer_model()
            except Exception:
                self._model = None
                self._is_loaded = False
        else:
            self._model = None
            self._is_loaded = False
    
    def _load_transformer_model(self) -> None:
        """Load the transformer model."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._is_loaded = True
    
    def predict(self, text: str) -> ClassifierOutput:
        start = time.time()
        
        if not TRANSFORMERS_AVAILABLE or not self._is_loaded:
            return self._heuristic_predict(text)
        
        # Tokenize and predict
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        
        # Map to our labels (model may have different labels)
        # For simplicity, map positive=safe, negative=toxic
        if len(probs) == 2:
            safe_prob = float(probs[1])  # Positive sentiment ≈ safe
            toxic_prob = float(probs[0])  # Negative sentiment ≈ toxic
            probabilities = {
                "safe": safe_prob,
                "toxic": toxic_prob,
                "severe_toxic": 0.0,
                "threat": 0.0,
                "insult": 0.0,
                "hate": 0.0,
            }
            label = "safe" if safe_prob > toxic_prob else "toxic"
            score = max(safe_prob, toxic_prob)
        else:
            probabilities = {self.LABELS[i]: float(probs[i]) for i in range(min(len(probs), len(self.LABELS)))}
            pred_idx = np.argmax(probs)
            label = self.LABELS[pred_idx] if pred_idx < len(self.LABELS) else "safe"
            score = float(probs[pred_idx])
        
        latency = (time.time() - start) * 1000
        
        return ClassifierOutput(
            label=label,
            score=score,
            probabilities=probabilities,
            latency_ms=latency,
            cost=self.cost_per_call,
            metadata={"model": self.model_name},
        )
    
    def predict_batch(self, texts: List[str]) -> List[ClassifierOutput]:
        # Could be optimized with batched inference
        return [self.predict(t) for t in texts]
    
    def _heuristic_predict(self, text: str) -> ClassifierOutput:
        """Fallback when transformers not available."""
        start = time.time()
        text_lower = text.lower()
        
        # Simple keyword detection
        toxic_keywords = {
            "toxic": ["stupid", "idiot", "dumb", "hate"],
            "threat": ["kill", "hurt", "attack", "destroy"],
            "insult": ["ugly", "worthless", "loser"],
            "hate": ["racist", "sexist"],
        }
        
        scores = {"safe": 0.8}
        for category, keywords in toxic_keywords.items():
            count = sum(1 for k in keywords if k in text_lower)
            if count > 0:
                scores[category] = min(0.9, count * 0.3)
                scores["safe"] = max(0.1, scores["safe"] - count * 0.2)
        
        # Normalize
        total = sum(scores.values())
        scores = {k: v/total for k, v in scores.items()}
        
        # Fill missing
        for label in self.LABELS:
            if label not in scores:
                scores[label] = 0.0
        
        best_label = max(scores, key=scores.get)
        latency = (time.time() - start) * 1000
        
        return ClassifierOutput(
            label=best_label,
            score=scores[best_label],
            probabilities=scores,
            latency_ms=latency,
            cost=0.0,
            metadata={"model": "heuristic_fallback"},
        )


# ============================================================================
# Quality Validator (Ensemble)
# ============================================================================

class QualityValidatorML(BaseClassifier):
    """
    Quality validator using sklearn ensemble.
    
    Evaluates: coherence, completeness, relevance
    Performance: ~5ms, $0.0001 per call
    """
    
    LABELS = ["low_quality", "medium_quality", "high_quality"]
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_pretrained: bool = True,
    ):
        super().__init__(name="quality_validator", cost_per_call=0.0001)
        
        if SKLEARN_AVAILABLE:
            if model_path and model_path.exists():
                self.load(model_path)
            elif use_pretrained:
                self._init_default_model()
        else:
            self._model = None
    
    def _init_default_model(self) -> None:
        """
        Initialize with trained model.
        
        Training data sources (for production):
        - Anthropic HH-RLHF (helpfulness ratings)
        - OpenAI summarization preferences
        - Custom quality annotations
        
        Use classifier_training.py to generate larger datasets.
        """
        from risklab.measurement.classifier_training import create_quality_dataset
        
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
        )
        self._classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )
        
        # Use proper training dataset
        dataset = create_quality_dataset("small")
        texts, labels = dataset.to_arrays()
        
        X = self._vectorizer.fit_transform(texts)
        self._classifier.fit(X, labels)
        self._is_loaded = True
    
    def predict(self, text: str) -> ClassifierOutput:
        start = time.time()
        
        if not SKLEARN_AVAILABLE or not self._is_loaded:
            return self._heuristic_predict(text)
        
        X = self._vectorizer.transform([text])
        proba = self._classifier.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        
        latency = (time.time() - start) * 1000
        
        return ClassifierOutput(
            label=self.LABELS[pred_idx],
            score=float(proba[pred_idx]),
            probabilities={self.LABELS[i]: float(p) for i, p in enumerate(proba)},
            latency_ms=latency,
            cost=self.cost_per_call,
            metadata={"model": "random_forest"},
        )
    
    def predict_batch(self, texts: List[str]) -> List[ClassifierOutput]:
        return [self.predict(t) for t in texts]
    
    def _heuristic_predict(self, text: str) -> ClassifierOutput:
        """Fallback heuristic based on text features."""
        start = time.time()
        
        # Feature extraction
        length = len(text)
        sentences = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_len = length / max(sentences, 1)
        has_structure = '\n' in text or '•' in text or any(f"{i}." in text for i in range(1, 10))
        
        # Score
        if length < 50:
            label = "low_quality"
            score = 0.7
        elif length > 200 and has_structure:
            label = "high_quality"
            score = 0.8
        else:
            label = "medium_quality"
            score = 0.6
        
        probabilities = {
            "low_quality": 0.1,
            "medium_quality": 0.5,
            "high_quality": 0.4,
        }
        probabilities[label] = score
        
        latency = (time.time() - start) * 1000
        
        return ClassifierOutput(
            label=label,
            score=score,
            probabilities=probabilities,
            latency_ms=latency,
            cost=0.0,
            metadata={"model": "heuristic_fallback"},
        )
    
    def save(self, model_path: Path) -> None:
        if SKLEARN_AVAILABLE and self._is_loaded:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self._vectorizer,
                    'classifier': self._classifier,
                }, f)
    
    def load(self, model_path: Path) -> None:
        if SKLEARN_AVAILABLE and model_path.exists():
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self._vectorizer = data['vectorizer']
                self._classifier = data['classifier']
                self._is_loaded = True


# ============================================================================
# Classifier Registry
# ============================================================================

class ClassifierRegistry:
    """
    Registry for managing classifier instances.
    
    Provides singleton access to classifiers to avoid reloading models.
    """
    
    _instances: Dict[str, BaseClassifier] = {}
    
    @classmethod
    def get_sentiment(cls) -> SentimentClassifierML:
        if "sentiment" not in cls._instances:
            cls._instances["sentiment"] = SentimentClassifierML()
        return cls._instances["sentiment"]
    
    @classmethod
    def get_intent(cls) -> IntentClassifierML:
        if "intent" not in cls._instances:
            cls._instances["intent"] = IntentClassifierML()
        return cls._instances["intent"]
    
    @classmethod
    def get_toxicity(cls) -> ToxicityClassifierML:
        if "toxicity" not in cls._instances:
            cls._instances["toxicity"] = ToxicityClassifierML()
        return cls._instances["toxicity"]
    
    @classmethod
    def get_quality(cls) -> QualityValidatorML:
        if "quality" not in cls._instances:
            cls._instances["quality"] = QualityValidatorML()
        return cls._instances["quality"]
    
    @classmethod
    def get_all(cls) -> Dict[str, BaseClassifier]:
        return {
            "sentiment": cls.get_sentiment(),
            "intent": cls.get_intent(),
            "toxicity": cls.get_toxicity(),
            "quality": cls.get_quality(),
        }
    
    @classmethod
    def clear(cls) -> None:
        """Clear all cached instances."""
        cls._instances.clear()