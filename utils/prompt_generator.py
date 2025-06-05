from typing import Optional
from models.character import Character
from models.scenario import Scenario

class PromptGenerator:
    @staticmethod
    def generate_character_prompt(character: Character) -> str:
        """Generate a system prompt for a character"""
        return character.generate_prompt_section()
    
    @staticmethod
    def generate_scenario_prompt(scenario: Scenario) -> str:
        """Generate a system prompt for a scenario"""
        return scenario.generate_prompt_section()
    
    @staticmethod
    def combine_prompts(
        character: Optional[Character] = None,
        scenario: Optional[Scenario] = None,
        additional_instructions: str = ""
    ) -> str:
        """Combine character and scenario prompts with optional instructions"""
        prompt_parts = []
        
        if character and scenario:
            prompt_parts.append(f"You are {character.name}, a {character.role} in {scenario.setting}.")
            prompt_parts.append(f"\n=== Character Details ===\n{character.generate_prompt_section()}")
            prompt_parts.append(f"\n=== World Context ===\n{scenario.generate_prompt_section()}")
        elif character:
            prompt_parts.append(character.generate_prompt_section())
        elif scenario:
            prompt_parts.append(scenario.generate_prompt_section())
        else:
            prompt_parts.append("You are a helpful AI assistant.")
        
        if additional_instructions:
            prompt_parts.append(f"\n=== Additional Instructions ===\n{additional_instructions}")
        
        return "\n\n".join(prompt_parts)
    
    @staticmethod
    def format_personality_traits(traits: dict) -> str:
        """Format personality traits into readable text"""
        descriptions = []
        for trait, value in traits.items():
            if value > 0.7:
                descriptions.append(f"very {trait}")
            elif value > 0.3:
                descriptions.append(f"moderately {trait}")
            else:
                descriptions.append(f"not very {trait}")
        return ", ".join(descriptions)
