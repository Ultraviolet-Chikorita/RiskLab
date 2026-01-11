"""
White-box instrumentation for deep model introspection.

Provides:
- Activation extraction at any layer
- Attention pattern analysis
- Linear probes for internal representations
- SAE (Sparse Autoencoder) integration for interpretability
"""

from typing import Optional, List, Dict, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from pydantic import BaseModel, Field


class LayerType(str, Enum):
    """Types of layers to instrument."""
    ATTENTION = "attention"
    MLP = "mlp"
    RESIDUAL = "residual"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"


@dataclass
class ActivationCapture:
    """Captured activation from a model layer."""
    layer_name: str
    layer_idx: int
    layer_type: LayerType
    shape: Tuple[int, ...]
    data: Any  # numpy array or torch tensor
    token_positions: Optional[List[int]] = None
    
    def to_numpy(self) -> np.ndarray:
        """Convert activation to numpy array."""
        if hasattr(self.data, 'cpu'):
            return self.data.cpu().numpy()
        return np.array(self.data)
    
    def mean_activation(self) -> float:
        """Get mean activation value."""
        arr = self.to_numpy()
        return float(np.mean(arr))
    
    def norm(self) -> float:
        """Get L2 norm of activation."""
        arr = self.to_numpy()
        return float(np.linalg.norm(arr))


@dataclass 
class AttentionPattern:
    """Captured attention pattern from a layer."""
    layer_idx: int
    head_idx: Optional[int]
    pattern: Any  # [seq_len, seq_len] or [num_heads, seq_len, seq_len]
    tokens: Optional[List[str]] = None
    
    def to_numpy(self) -> np.ndarray:
        if hasattr(self.pattern, 'cpu'):
            return self.pattern.cpu().numpy()
        return np.array(self.pattern)
    
    def get_attention_to_token(self, token_idx: int) -> np.ndarray:
        """Get attention weights flowing to a specific token."""
        arr = self.to_numpy()
        if len(arr.shape) == 3:  # [heads, seq, seq]
            return arr[:, :, token_idx]
        return arr[:, token_idx]
    
    def get_attention_from_token(self, token_idx: int) -> np.ndarray:
        """Get attention weights flowing from a specific token."""
        arr = self.to_numpy()
        if len(arr.shape) == 3:
            return arr[:, token_idx, :]
        return arr[token_idx, :]


class WhiteBoxConfig(BaseModel):
    """Configuration for white-box instrumentation."""
    capture_activations: bool = True
    capture_attention: bool = True
    
    # Layer selection
    layers_to_capture: Optional[List[int]] = None  # None = all layers
    layer_types: List[LayerType] = Field(default_factory=lambda: [LayerType.RESIDUAL])
    
    # Attention settings
    aggregate_heads: bool = False  # Average across attention heads
    
    # Memory management
    keep_on_gpu: bool = False
    max_seq_length: Optional[int] = 512  # Truncate for memory
    
    # Probe settings
    probe_layer_indices: List[int] = Field(default_factory=list)
    
    # SAE settings
    use_sae: bool = False
    sae_model_path: Optional[str] = None


class ActivationHook:
    """Hook for capturing activations during forward pass."""
    
    def __init__(
        self,
        layer_name: str,
        layer_idx: int,
        layer_type: LayerType,
        keep_on_gpu: bool = False,
    ):
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.keep_on_gpu = keep_on_gpu
        self.captured: Optional[ActivationCapture] = None
        self._handle = None
    
    def __call__(self, module, input, output):
        """Hook function called during forward pass."""
        # Handle different output formats
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output
        
        # Move to CPU if needed
        if not self.keep_on_gpu and hasattr(activation, 'cpu'):
            activation = activation.detach().cpu()
        elif hasattr(activation, 'detach'):
            activation = activation.detach()
        
        self.captured = ActivationCapture(
            layer_name=self.layer_name,
            layer_idx=self.layer_idx,
            layer_type=self.layer_type,
            shape=tuple(activation.shape),
            data=activation,
        )
    
    def register(self, module) -> None:
        """Register hook on a module."""
        self._handle = module.register_forward_hook(self)
    
    def remove(self) -> None:
        """Remove the hook."""
        if self._handle:
            self._handle.remove()
            self._handle = None


class AttentionHook:
    """Hook for capturing attention patterns."""
    
    def __init__(
        self,
        layer_idx: int,
        aggregate_heads: bool = False,
        keep_on_gpu: bool = False,
    ):
        self.layer_idx = layer_idx
        self.aggregate_heads = aggregate_heads
        self.keep_on_gpu = keep_on_gpu
        self.captured: Optional[AttentionPattern] = None
        self._handle = None
    
    def __call__(self, module, input, output):
        """Hook to capture attention weights."""
        # Attention output format varies by model
        # Common formats: (attn_output, attn_weights) or just attn_output
        attn_weights = None
        
        if isinstance(output, tuple) and len(output) >= 2:
            # Second element is often attention weights
            potential_weights = output[1]
            if potential_weights is not None:
                attn_weights = potential_weights
        
        if attn_weights is None:
            # Try to get from module's stored attention
            if hasattr(module, 'attn_weights'):
                attn_weights = module.attn_weights
        
        if attn_weights is not None:
            if not self.keep_on_gpu and hasattr(attn_weights, 'cpu'):
                attn_weights = attn_weights.detach().cpu()
            elif hasattr(attn_weights, 'detach'):
                attn_weights = attn_weights.detach()
            
            if self.aggregate_heads and len(attn_weights.shape) == 4:
                # [batch, heads, seq, seq] -> [batch, seq, seq]
                attn_weights = attn_weights.mean(dim=1)
            
            self.captured = AttentionPattern(
                layer_idx=self.layer_idx,
                head_idx=None if self.aggregate_heads else 0,
                pattern=attn_weights[0] if attn_weights.shape[0] == 1 else attn_weights,
            )
    
    def register(self, module) -> None:
        self._handle = module.register_forward_hook(self)
    
    def remove(self) -> None:
        if self._handle:
            self._handle.remove()
            self._handle = None


class WhiteBoxInstrumenter:
    """
    Instruments a HuggingFace model for white-box analysis.
    
    Supports:
    - Activation extraction at specified layers
    - Attention pattern capture
    - Integration with linear probes
    - SAE-based feature extraction
    """
    
    def __init__(self, model, tokenizer, config: Optional[WhiteBoxConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or WhiteBoxConfig()
        
        self._activation_hooks: List[ActivationHook] = []
        self._attention_hooks: List[AttentionHook] = []
        self._probes: Dict[int, 'LinearProbe'] = {}
        self._sae: Optional['SparseAutoencoder'] = None
        
        self._model_info = self._analyze_model_structure()
    
    def _analyze_model_structure(self) -> Dict[str, Any]:
        """Analyze model architecture to find instrumentable layers."""
        info = {
            "num_layers": 0,
            "layer_modules": [],
            "attention_modules": [],
            "mlp_modules": [],
            "hidden_size": None,
        }
        
        # Common layer patterns for different model architectures
        layer_patterns = [
            "model.layers",  # Llama, Mistral
            "transformer.h",  # GPT-2, GPT-Neo
            "model.decoder.layers",  # BART, T5
            "encoder.layer",  # BERT
            "transformer.blocks",  # Falcon
        ]
        
        for pattern in layer_patterns:
            parts = pattern.split(".")
            module = self.model
            try:
                for part in parts:
                    module = getattr(module, part)
                info["num_layers"] = len(module)
                info["layer_modules"] = list(module)
                break
            except (AttributeError, TypeError):
                continue
        
        # Get hidden size from config
        if hasattr(self.model, 'config'):
            config = self.model.config
            for attr in ['hidden_size', 'd_model', 'n_embd']:
                if hasattr(config, attr):
                    info["hidden_size"] = getattr(config, attr)
                    break
        
        return info
    
    def setup_hooks(self) -> None:
        """Set up all instrumentation hooks."""
        self.clear_hooks()
        
        layers_to_capture = self.config.layers_to_capture
        if layers_to_capture is None:
            layers_to_capture = list(range(self._model_info["num_layers"]))
        
        for layer_idx in layers_to_capture:
            if layer_idx >= len(self._model_info["layer_modules"]):
                continue
            
            layer = self._model_info["layer_modules"][layer_idx]
            
            # Activation hooks
            if self.config.capture_activations:
                if LayerType.RESIDUAL in self.config.layer_types:
                    hook = ActivationHook(
                        f"layer_{layer_idx}_residual",
                        layer_idx,
                        LayerType.RESIDUAL,
                        self.config.keep_on_gpu,
                    )
                    hook.register(layer)
                    self._activation_hooks.append(hook)
                
                if LayerType.MLP in self.config.layer_types:
                    # Find MLP submodule
                    mlp = None
                    for name in ['mlp', 'feed_forward', 'ffn', 'fc']:
                        if hasattr(layer, name):
                            mlp = getattr(layer, name)
                            break
                    
                    if mlp:
                        hook = ActivationHook(
                            f"layer_{layer_idx}_mlp",
                            layer_idx,
                            LayerType.MLP,
                            self.config.keep_on_gpu,
                        )
                        hook.register(mlp)
                        self._activation_hooks.append(hook)
            
            # Attention hooks
            if self.config.capture_attention:
                attn = None
                for name in ['self_attn', 'attention', 'attn']:
                    if hasattr(layer, name):
                        attn = getattr(layer, name)
                        break
                
                if attn:
                    hook = AttentionHook(
                        layer_idx,
                        self.config.aggregate_heads,
                        self.config.keep_on_gpu,
                    )
                    hook.register(attn)
                    self._attention_hooks.append(hook)
    
    def clear_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self._activation_hooks:
            hook.remove()
        for hook in self._attention_hooks:
            hook.remove()
        self._activation_hooks.clear()
        self._attention_hooks.clear()
    
    def get_activations(self) -> List[ActivationCapture]:
        """Get all captured activations."""
        return [h.captured for h in self._activation_hooks if h.captured is not None]
    
    def get_attention_patterns(self) -> List[AttentionPattern]:
        """Get all captured attention patterns."""
        return [h.captured for h in self._attention_hooks if h.captured is not None]
    
    def get_layer_activation(self, layer_idx: int) -> Optional[ActivationCapture]:
        """Get activation for a specific layer."""
        for hook in self._activation_hooks:
            if hook.captured and hook.layer_idx == layer_idx:
                return hook.captured
        return None
    
    def add_probe(self, layer_idx: int, probe: 'LinearProbe') -> None:
        """Add a linear probe for a specific layer."""
        self._probes[layer_idx] = probe
    
    def run_probes(self) -> Dict[int, Dict[str, float]]:
        """Run all registered probes on captured activations."""
        results = {}
        for layer_idx, probe in self._probes.items():
            activation = self.get_layer_activation(layer_idx)
            if activation:
                results[layer_idx] = probe.predict(activation)
        return results
    
    def set_sae(self, sae: 'SparseAutoencoder') -> None:
        """Set SAE for feature extraction."""
        self._sae = sae
    
    def extract_sae_features(self, layer_idx: int) -> Optional[Dict[str, Any]]:
        """Extract SAE features from a layer's activations."""
        if self._sae is None:
            return None
        
        activation = self.get_layer_activation(layer_idx)
        if activation is None:
            return None
        
        return self._sae.encode(activation.to_numpy())
    
    def __enter__(self):
        self.setup_hooks()
        return self
    
    def __exit__(self, *args):
        self.clear_hooks()


class LinearProbe:
    """
    Linear probe for analyzing internal representations.
    
    Used to predict properties (e.g., sentiment, factuality) from activations.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        probe_type: str = "binary",  # binary, multiclass, regression
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.probe_type = probe_type
        
        # Initialize weights
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self._is_trained = False
    
    def train(
        self,
        activations: List[np.ndarray],
        labels: List[Any],
        epochs: int = 100,
        learning_rate: float = 0.01,
    ) -> Dict[str, float]:
        """Train the probe on labeled activations."""
        X = np.stack([a.reshape(-1)[:self.input_dim] for a in activations])
        
        if self.probe_type == "binary":
            y = np.array(labels, dtype=np.float32).reshape(-1, 1)
        else:
            y = np.array(labels)
        
        # Simple gradient descent
        self.weights = np.random.randn(self.input_dim, self.output_dim) * 0.01
        self.bias = np.zeros(self.output_dim)
        
        losses = []
        for epoch in range(epochs):
            # Forward pass
            logits = X @ self.weights + self.bias
            
            if self.probe_type == "binary":
                preds = 1 / (1 + np.exp(-logits))  # Sigmoid
                loss = -np.mean(y * np.log(preds + 1e-8) + (1 - y) * np.log(1 - preds + 1e-8))
                grad = (preds - y) / len(y)
            else:
                loss = np.mean((logits - y) ** 2)
                grad = 2 * (logits - y) / len(y)
            
            # Backward pass
            self.weights -= learning_rate * (X.T @ grad)
            self.bias -= learning_rate * np.mean(grad, axis=0)
            
            losses.append(loss)
        
        self._is_trained = True
        return {"final_loss": losses[-1], "epochs": epochs}
    
    def predict(self, activation: Union[ActivationCapture, np.ndarray]) -> Dict[str, float]:
        """Run prediction on an activation."""
        if not self._is_trained:
            raise ValueError("Probe not trained")
        
        if isinstance(activation, ActivationCapture):
            x = activation.to_numpy()
        else:
            x = activation
        
        # Flatten and truncate/pad to input_dim
        x = x.reshape(-1)
        if len(x) > self.input_dim:
            x = x[:self.input_dim]
        elif len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        
        logits = x @ self.weights + self.bias
        
        if self.probe_type == "binary":
            prob = 1 / (1 + np.exp(-logits[0]))
            return {"probability": float(prob), "prediction": int(prob > 0.5)}
        else:
            return {"prediction": float(logits[0])}
    
    def save(self, path: str) -> None:
        """Save probe weights."""
        np.savez(path, weights=self.weights, bias=self.bias, 
                 input_dim=self.input_dim, output_dim=self.output_dim,
                 probe_type=self.probe_type)
    
    @classmethod
    def load(cls, path: str) -> 'LinearProbe':
        """Load probe from file."""
        data = np.load(path)
        probe = cls(
            input_dim=int(data['input_dim']),
            output_dim=int(data['output_dim']),
            probe_type=str(data['probe_type']),
        )
        probe.weights = data['weights']
        probe.bias = data['bias']
        probe._is_trained = True
        return probe


class SparseAutoencoder:
    """
    Sparse Autoencoder for interpretable feature extraction.
    
    Learns a sparse, overcomplete representation of activations
    that can reveal interpretable features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_coef: float = 0.01,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coef = sparsity_coef
        
        # Initialize weights
        self.encoder_weights: Optional[np.ndarray] = None
        self.encoder_bias: Optional[np.ndarray] = None
        self.decoder_weights: Optional[np.ndarray] = None
        self.decoder_bias: Optional[np.ndarray] = None
        
        self._is_trained = False
        self._feature_labels: Dict[int, str] = {}
    
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization."""
        scale = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        self.encoder_weights = np.random.randn(self.input_dim, self.hidden_dim) * scale
        self.encoder_bias = np.zeros(self.hidden_dim)
        self.decoder_weights = np.random.randn(self.hidden_dim, self.input_dim) * scale
        self.decoder_bias = np.zeros(self.input_dim)
    
    def train(
        self,
        activations: List[np.ndarray],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> Dict[str, Any]:
        """Train the SAE on activations."""
        self._initialize_weights()
        
        X = np.stack([a.reshape(-1)[:self.input_dim] for a in activations])
        n_samples = len(X)
        
        losses = []
        sparsities = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            epoch_sparsity = 0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                batch = X[batch_idx]
                
                # Forward pass
                hidden = self._encode(batch)
                recon = self._decode(hidden)
                
                # Losses
                recon_loss = np.mean((batch - recon) ** 2)
                sparsity_loss = self.sparsity_coef * np.mean(np.abs(hidden))
                total_loss = recon_loss + sparsity_loss
                
                # Backward pass (simplified gradient descent)
                recon_grad = 2 * (recon - batch) / len(batch)
                
                # Decoder gradients
                d_decoder_weights = hidden.T @ recon_grad
                d_decoder_bias = np.mean(recon_grad, axis=0)
                
                # Encoder gradients
                hidden_grad = recon_grad @ self.decoder_weights.T
                hidden_grad += self.sparsity_coef * np.sign(hidden) / len(batch)
                hidden_grad *= (hidden > 0).astype(float)  # ReLU derivative
                
                d_encoder_weights = batch.T @ hidden_grad
                d_encoder_bias = np.mean(hidden_grad, axis=0)
                
                # Update weights
                self.encoder_weights -= learning_rate * d_encoder_weights
                self.encoder_bias -= learning_rate * d_encoder_bias
                self.decoder_weights -= learning_rate * d_decoder_weights
                self.decoder_bias -= learning_rate * d_decoder_bias
                
                epoch_loss += total_loss
                epoch_sparsity += np.mean(hidden > 0)
            
            losses.append(epoch_loss / (n_samples // batch_size))
            sparsities.append(epoch_sparsity / (n_samples // batch_size))
        
        self._is_trained = True
        
        return {
            "final_loss": losses[-1],
            "final_sparsity": sparsities[-1],
            "epochs": epochs,
        }
    
    def _encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to hidden representation."""
        hidden = x @ self.encoder_weights + self.encoder_bias
        return np.maximum(0, hidden)  # ReLU activation
    
    def _decode(self, hidden: np.ndarray) -> np.ndarray:
        """Decode hidden representation to reconstruction."""
        return hidden @ self.decoder_weights + self.decoder_bias
    
    def encode(self, activation: Union[np.ndarray, ActivationCapture]) -> Dict[str, Any]:
        """Encode an activation and return feature analysis."""
        if not self._is_trained:
            raise ValueError("SAE not trained")
        
        if isinstance(activation, ActivationCapture):
            x = activation.to_numpy()
        else:
            x = activation
        
        x = x.reshape(-1)[:self.input_dim]
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        
        x = x.reshape(1, -1)
        hidden = self._encode(x)[0]
        
        # Find active features
        active_indices = np.where(hidden > 0)[0]
        active_values = hidden[active_indices]
        
        # Sort by activation strength
        sorted_idx = np.argsort(-active_values)
        active_indices = active_indices[sorted_idx]
        active_values = active_values[sorted_idx]
        
        # Get feature labels if available
        active_features = []
        for idx, val in zip(active_indices[:20], active_values[:20]):
            feature = {
                "index": int(idx),
                "activation": float(val),
                "label": self._feature_labels.get(int(idx), f"feature_{idx}"),
            }
            active_features.append(feature)
        
        return {
            "num_active_features": len(active_indices),
            "sparsity": len(active_indices) / self.hidden_dim,
            "top_features": active_features,
            "total_activation": float(np.sum(hidden)),
            "raw_encoding": hidden,
        }
    
    def label_feature(self, feature_idx: int, label: str) -> None:
        """Assign a human-readable label to a feature."""
        self._feature_labels[feature_idx] = label
    
    def get_feature_direction(self, feature_idx: int) -> np.ndarray:
        """Get the decoder direction for a feature (what it represents in input space)."""
        if not self._is_trained:
            raise ValueError("SAE not trained")
        return self.decoder_weights[feature_idx]
    
    def save(self, path: str) -> None:
        """Save SAE weights and labels."""
        np.savez(
            path,
            encoder_weights=self.encoder_weights,
            encoder_bias=self.encoder_bias,
            decoder_weights=self.decoder_weights,
            decoder_bias=self.decoder_bias,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            sparsity_coef=self.sparsity_coef,
            feature_labels=str(self._feature_labels),
        )
    
    @classmethod
    def load(cls, path: str) -> 'SparseAutoencoder':
        """Load SAE from file."""
        data = np.load(path, allow_pickle=True)
        sae = cls(
            input_dim=int(data['input_dim']),
            hidden_dim=int(data['hidden_dim']),
            sparsity_coef=float(data['sparsity_coef']),
        )
        sae.encoder_weights = data['encoder_weights']
        sae.encoder_bias = data['encoder_bias']
        sae.decoder_weights = data['decoder_weights']
        sae.decoder_bias = data['decoder_bias']
        sae._is_trained = True
        
        # Parse feature labels
        try:
            sae._feature_labels = eval(str(data['feature_labels']))
        except:
            pass
        
        return sae


class WhiteBoxAnalysisResult(BaseModel):
    """Result of white-box analysis on a generation."""
    
    # Activation statistics
    layer_activation_norms: Dict[int, float] = Field(default_factory=dict)
    layer_activation_means: Dict[int, float] = Field(default_factory=dict)
    
    # Attention analysis
    attention_entropy: Dict[int, float] = Field(default_factory=dict)
    attention_to_special_tokens: Dict[int, float] = Field(default_factory=dict)
    
    # Probe results
    probe_predictions: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # SAE features
    sae_features: Optional[Dict[str, Any]] = None
    top_active_features: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Anomaly detection
    anomaly_scores: Dict[str, float] = Field(default_factory=dict)
    
    def to_summary(self) -> Dict[str, Any]:
        """Get summary for reporting."""
        return {
            "num_layers_analyzed": len(self.layer_activation_norms),
            "mean_activation_norm": np.mean(list(self.layer_activation_norms.values())) if self.layer_activation_norms else 0,
            "mean_attention_entropy": np.mean(list(self.attention_entropy.values())) if self.attention_entropy else 0,
            "probe_predictions": self.probe_predictions,
            "num_active_sae_features": len(self.top_active_features),
            "anomaly_scores": self.anomaly_scores,
        }


def analyze_attention_patterns(patterns: List[AttentionPattern]) -> Dict[str, Any]:
    """Analyze attention patterns for interesting behaviors."""
    results = {}
    
    for pattern in patterns:
        arr = pattern.to_numpy()
        layer_idx = pattern.layer_idx
        
        # Compute attention entropy (how focused vs distributed)
        # Higher entropy = more distributed attention
        if len(arr.shape) == 2:
            # Single head or aggregated
            entropy = -np.sum(arr * np.log(arr + 1e-10), axis=-1)
            results[f"layer_{layer_idx}_entropy"] = float(np.mean(entropy))
        else:
            # Multiple heads
            entropies = []
            for head in range(arr.shape[0]):
                head_pattern = arr[head]
                entropy = -np.sum(head_pattern * np.log(head_pattern + 1e-10), axis=-1)
                entropies.append(np.mean(entropy))
            results[f"layer_{layer_idx}_entropy"] = float(np.mean(entropies))
        
        # Check for attention sinks (tokens that receive disproportionate attention)
        mean_attention_received = np.mean(arr, axis=-2)  # Average over source positions
        max_attention = np.max(mean_attention_received)
        results[f"layer_{layer_idx}_max_attention_sink"] = float(max_attention)
    
    return results


def compute_activation_anomaly_score(
    activation: ActivationCapture,
    reference_mean: Optional[np.ndarray] = None,
    reference_std: Optional[np.ndarray] = None,
) -> float:
    """
    Compute anomaly score for an activation relative to reference distribution.
    
    Higher scores indicate more unusual activations.
    """
    arr = activation.to_numpy().reshape(-1)
    
    if reference_mean is None or reference_std is None:
        # Use simple statistics
        return float(np.std(arr) / (np.mean(np.abs(arr)) + 1e-8))
    
    # Z-score based anomaly
    z_scores = np.abs(arr - reference_mean) / (reference_std + 1e-8)
    return float(np.mean(z_scores))
