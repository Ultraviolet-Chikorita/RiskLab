"""
Extended Scenario Library with Multi-Message Episodes.

Extends the default scenario library to include multi-message conversations
designed to elicit and track bias across conversation turns.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

from risklab.scenarios.library import ScenarioLibrary, load_default_scenarios
from risklab.scenarios.multi_message_episodes import create_multi_message_episodes, MultiMessageEpisode, HiddenAgenda
from risklab.scenarios.system_prompt_generator import SystemPromptGenerator, create_authority_bias_test, create_profit_bias_test
from risklab.scenarios.framing import FramingType


def load_extended_scenarios() -> ScenarioLibrary:
    """Load the extended scenario library including multi-message episodes."""
    
    # Load base scenarios
    library = load_default_scenarios()
    
    # Add multi-message episodes
    multi_episodes = create_multi_message_episodes()
    for episode in multi_episodes:
        library.add(episode)
    
    return library


def create_bias_test_library() -> ScenarioLibrary:
    """Create a library specifically focused on bias testing."""
    
    library = ScenarioLibrary()
    
    # Add multi-message episodes with system prompts
    generator = SystemPromptGenerator()
    
    # Create episodes with different bias configurations
    multi_episodes = create_multi_message_episodes()
    
    for episode in multi_episodes:
        # Generate system prompts based on episode category
        if episode.category == "authority_bias":
            system_prompts = create_authority_bias_test()
        elif episode.category == "financial_bias":
            system_prompts = create_profit_bias_test()
        else:
            # Default to user-pleasing spectrum
            from risklab.scenarios.system_prompt_generator import create_user_pleasing_spectrum
            system_prompts = create_user_pleasing_spectrum()
        
        # Apply system prompts to episode
        for framing, prompt_obj in system_prompts.items():
            episode.system_prompts[framing] = prompt_obj.prompt
            if prompt_obj.agenda:
                episode.hidden_agendas[framing] = prompt_obj.agenda
        
        library.add(episode)
    
    return library


def create_progressive_bias_library() -> ScenarioLibrary:
    """Create a library with progressively more intense bias scenarios."""
    
    library = ScenarioLibrary()
    
    generator = SystemPromptGenerator(seed=42)  # For reproducibility
    
    # Get base multi-message episodes
    base_episodes = create_multi_message_episodes()
    
    # Create variants with different bias intensities
    from risklab.scenarios.system_prompt_generator import BiasIntensity
    
    for base_episode in base_episodes:
        # Create three variants: subtle, moderate, overt
        for intensity in [BiasIntensity.SUBTLE, BiasIntensity.MODERATE, BiasIntensity.OVERT]:
            
            # Clone the episode
            episode = MultiMessageEpisode(
                name=f"{base_episode.name} ({intensity.value.title()} Bias)",
                description=base_episode.description,
                domain=base_episode.domain,
                stakes_level=base_episode.stakes_level,
                messages=base_episode.messages.copy(),
                expected_biases=base_episode.expected_biases.copy(),
                pressure_points=base_episode.pressure_points.copy(),
                tags=base_episode.tags + [f"bias_intensity_{intensity.value}"],
                category=base_episode.category
            )
            
            # Generate system prompts with specified intensity
            if base_episode.category == "authority_bias":
                agenda = generator.generate_prompt(
                    agenda=HiddenAgenda.AUTHORITY_DEFERENCE,
                    intensity=intensity,
                    framing_type=FramingType.OVERSIGHT
                )
                episode.system_prompts[FramingType.OVERSIGHT] = agenda.prompt
                episode.hidden_agendas[FramingType.OVERSIGHT] = agenda.agenda
                
            elif base_episode.category == "financial_bias":
                agenda = generator.generate_prompt(
                    agenda=HiddenAgenda.PROFIT_MAXIMIZATION,
                    intensity=intensity,
                    framing_type=FramingType.EVALUATED
                )
                episode.system_prompts[FramingType.EVALUATED] = agenda.prompt
                episode.hidden_agendas[FramingType.EVALUATED] = agenda.agenda
            
            # Add user-pleasing bias to evaluated framing
            agenda = generator.generate_prompt(
                agenda=HiddenAgenda.USER_PLEASING,
                intensity=intensity,
                framing_type=FramingType.EVALUATED
            )
            episode.system_prompts[FramingType.EVALUATED] = agenda.prompt
            episode.hidden_agendas[FramingType.EVALUATED] = agenda.agenda
            
            library.add(episode)
    
    return library


def save_extended_library(library: ScenarioLibrary, path: Optional[Path] = None):
    """Save the extended library to disk."""
    
    if path is None:
        path = Path("extended_scenarios.json")
    
    # Convert to serializable format
    data = {
        "name": library.name,
        "description": library.description,
        "episodes": [episode.model_dump() for episode in library.episodes],
        "metadata": library.metadata
    }
    
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"Extended library saved to {path}")


if __name__ == "__main__":
    # Example usage
    print("Loading extended scenarios...")
    library = load_extended_scenarios()
    print(f"Loaded {len(library.episodes)} episodes")
    
    # Save extended library
    save_extended_library(library)
    
    # Create bias test library
    bias_library = create_bias_test_library()
    save_extended_library(bias_library, Path("bias_test_scenarios.json"))
    
    print("Extended libraries created successfully!")
