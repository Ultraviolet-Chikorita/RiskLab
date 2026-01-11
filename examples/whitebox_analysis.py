"""
White-box analysis example for the Risk-Conditioned AI Evaluation Lab.

This example demonstrates how to:
1. Use white-box instrumentation with HuggingFace models
2. Extract and analyze activations
3. Capture attention patterns
4. Train and use linear probes
5. Use SAEs for interpretable feature extraction
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from risklab.config import LabConfig, ModelProviderType, InstrumentationMode, ModelConfig, APIConfig
from risklab.models import (
    load_model,
    WhiteBoxConfig,
    WhiteBoxInstrumenter,
    LinearProbe,
    SparseAutoencoder,
)
from risklab.measurement import (
    WhiteBoxMetrics,
    WhiteBoxBehaviorAnalyzer,
    WhiteBoxAnalyzerFactory,
    compute_whitebox_manipulation_score,
)


async def basic_whitebox_example():
    """Example: Basic white-box activation capture."""
    print("=" * 60)
    print("Example 1: Basic White-Box Activation Capture")
    print("=" * 60)
    
    # Load model with white-box instrumentation
    runtime = load_model(
        provider=ModelProviderType.HUGGINGFACE,
        model_name="gpt2",  # Small model for demo
        instrumentation_mode=InstrumentationMode.WHITE_BOX,
        device_map="cpu",  # Use CPU for demo
    )
    
    prompt = "The capital of France is"
    
    print(f"\nPrompt: {prompt}")
    print("Generating with white-box instrumentation...")
    
    result = await runtime.generate(prompt, max_tokens=20)
    
    print(f"\nResponse: {result.text}")
    
    # Check white-box data
    if result.activations:
        print(f"\n--- Activation Data ---")
        print(f"Number of layers captured: {result.activations.get('num_layers', 0)}")
        
        for layer in result.activations.get('layers', [])[:5]:
            print(f"  Layer {layer['layer_idx']}: norm={layer['norm']:.4f}, mean={layer['mean']:.4f}")
    
    if result.attention_patterns:
        print(f"\n--- Attention Data ---")
        print(f"Number of layers: {result.attention_patterns.get('num_layers', 0)}")
        
        for key, value in result.attention_patterns.items():
            if 'entropy' in key:
                print(f"  {key}: {value:.4f}")
    
    return result


async def probe_training_example():
    """Example: Training and using linear probes."""
    print("\n" + "=" * 60)
    print("Example 2: Linear Probe Training")
    print("=" * 60)
    
    # Create a probe
    hidden_size = 768  # GPT-2 hidden size
    probe = LinearProbe(input_dim=hidden_size, output_dim=1, probe_type="binary")
    
    print(f"\nCreated probe: input_dim={hidden_size}, type=binary")
    
    # Simulate training data (in practice, use real activations)
    print("\nSimulating training data...")
    np.random.seed(42)
    
    # Create fake "honest" and "deceptive" activations
    honest_activations = [np.random.randn(hidden_size) * 0.5 for _ in range(50)]
    deceptive_activations = [np.random.randn(hidden_size) * 0.5 + 0.3 for _ in range(50)]
    
    all_activations = honest_activations + deceptive_activations
    labels = [0] * 50 + [1] * 50  # 0 = honest, 1 = deceptive
    
    # Train the probe
    print("Training probe...")
    train_result = probe.train(all_activations, labels, epochs=100, learning_rate=0.01)
    
    print(f"Training complete: final_loss={train_result['final_loss']:.4f}")
    
    # Test prediction
    test_activation = np.random.randn(hidden_size) * 0.5 + 0.2
    prediction = probe.predict(test_activation)
    
    print(f"\nTest prediction: probability={prediction['probability']:.4f}, prediction={prediction['prediction']}")
    
    # Save probe
    probe_path = Path("./outputs/test_probe.npz")
    probe_path.parent.mkdir(exist_ok=True)
    probe.save(str(probe_path))
    print(f"\nProbe saved to: {probe_path}")
    
    return probe


async def sae_example():
    """Example: Sparse Autoencoder for feature extraction."""
    print("\n" + "=" * 60)
    print("Example 3: Sparse Autoencoder Features")
    print("=" * 60)
    
    # Create SAE
    input_dim = 768
    hidden_dim = 2048  # Overcomplete representation
    
    sae = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity_coef=0.01,
    )
    
    print(f"\nCreated SAE: input={input_dim}, hidden={hidden_dim}")
    
    # Simulate training data
    print("\nSimulating training activations...")
    np.random.seed(42)
    train_activations = [np.random.randn(input_dim) for _ in range(200)]
    
    # Train SAE
    print("Training SAE (this may take a moment)...")
    train_result = sae.train(train_activations, epochs=50, batch_size=32)
    
    print(f"Training complete:")
    print(f"  Final loss: {train_result['final_loss']:.4f}")
    print(f"  Final sparsity: {train_result['final_sparsity']:.4f}")
    
    # Encode a test activation
    test_activation = np.random.randn(input_dim)
    encoding = sae.encode(test_activation)
    
    print(f"\n--- SAE Encoding ---")
    print(f"Active features: {encoding['num_active_features']} / {hidden_dim}")
    print(f"Sparsity: {encoding['sparsity']:.4f}")
    print(f"Top features:")
    for feat in encoding['top_features'][:5]:
        print(f"  Feature {feat['index']}: activation={feat['activation']:.4f}")
    
    # Label some features (in practice, done through analysis)
    sae.label_feature(encoding['top_features'][0]['index'], "unknown_concept_1")
    
    return sae


async def full_whitebox_analysis():
    """Example: Full white-box behavioral analysis."""
    print("\n" + "=" * 60)
    print("Example 4: Full White-Box Behavioral Analysis")
    print("=" * 60)
    
    # Create analyzer with probes
    hidden_size = 768
    analyzer = WhiteBoxAnalyzerFactory.create_with_probes(hidden_size)
    
    print("\nCreated WhiteBoxBehaviorAnalyzer with:")
    print(f"  - Deception probe")
    print(f"  - Sycophancy probe")
    print(f"  - Uncertainty probe")
    
    # Create mock white-box metrics (in practice, from real model)
    mock_metrics = WhiteBoxMetrics(
        activation_anomaly_score=1.5,
        layer_activation_variance=0.3,
        late_layer_activation_spike=False,
        attention_entropy_mean=2.1,
        attention_sink_score=0.15,
        internal_consistency_score=0.85,
        num_active_features=150,
        feature_sparsity=0.07,
    )
    
    # Compute manipulation score
    manipulation_score = compute_whitebox_manipulation_score(mock_metrics)
    
    print(f"\n--- Analysis Results ---")
    print(f"Activation anomaly: {mock_metrics.activation_anomaly_score:.3f}")
    print(f"Attention entropy: {mock_metrics.attention_entropy_mean:.3f}")
    print(f"Internal consistency: {mock_metrics.internal_consistency_score:.3f}")
    print(f"Active SAE features: {mock_metrics.num_active_features}")
    print(f"\nOverall manipulation score: {manipulation_score:.3f}")
    
    # Interpretation
    if manipulation_score < 0.3:
        print("Interpretation: LOW manipulation risk")
    elif manipulation_score < 0.6:
        print("Interpretation: MODERATE manipulation risk")
    else:
        print("Interpretation: HIGH manipulation risk")


async def instrumenter_demo():
    """Example: Direct use of WhiteBoxInstrumenter."""
    print("\n" + "=" * 60)
    print("Example 5: Direct Instrumenter Usage")
    print("=" * 60)
    
    print("\nThis example requires a loaded HuggingFace model.")
    print("The WhiteBoxInstrumenter provides fine-grained control over:")
    print("  - Which layers to capture")
    print("  - Activation vs attention capture")
    print("  - Memory management (GPU vs CPU)")
    print("  - Hook lifecycle management")
    
    print("\nExample usage pattern:")
    print("""
    from risklab.models.whitebox import WhiteBoxInstrumenter, WhiteBoxConfig
    
    config = WhiteBoxConfig(
        capture_activations=True,
        capture_attention=True,
        layers_to_capture=[0, 6, 11],  # Specific layers
        aggregate_heads=True,  # Average attention heads
    )
    
    instrumenter = WhiteBoxInstrumenter(model, tokenizer, config)
    
    with instrumenter:  # Automatically sets up and clears hooks
        output = model.generate(inputs)
        
        activations = instrumenter.get_activations()
        attention = instrumenter.get_attention_patterns()
        
        for act in activations:
            print(f"Layer {act.layer_idx}: norm={act.norm():.4f}")
    """)


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Risk-Conditioned AI Evaluation Lab - White-Box Examples")
    print("=" * 60)
    
    # Check for HuggingFace availability
    try:
        import torch
        import transformers
        has_torch = True
    except ImportError:
        has_torch = False
        print("\n[!] PyTorch/Transformers not installed.")
        print("    Install with: pip install torch transformers")
        print("    Running demo-only examples...\n")
    
    if has_torch:
        try:
            await basic_whitebox_example()
        except Exception as e:
            print(f"\n[!] Basic example failed: {e}")
            print("    This may require GPU or more memory.")
    
    # These examples work without actual models
    await probe_training_example()
    await sae_example()
    await full_whitebox_analysis()
    await instrumenter_demo()
    
    print("\n" + "=" * 60)
    print("White-box examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
