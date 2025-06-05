from pydantic import BaseModel, Field
from typing import Optional
from .character import Character
from .scenario import Scenario

class PromptPack(BaseModel):
    name: str = Field(..., description="Name of the prompt pack")
    description: str = Field(
        default="",
        description="Description of the prompt pack"
    )
    character: Optional[Character] = Field(
        default=None,
        description="Character for this prompt pack"
    )
    scenario: Optional[Scenario] = Field(
        default=None,
        description="Scenario for this prompt pack"
    )

    def generate_system_prompt(self) -> str:
        """Generate the complete system prompt from character and scenario"""
        prompt_parts = []
        
        if self.character and self.scenario:
            # Full character + scenario prompt
            prompt_parts.append(f"You are {self.character.name}, a {self.character.age}-year-old {self.character.gender} {self.character.role} in {self.scenario.setting} during {self.scenario.time_period}.")
            
            # Add personality description
            personality_desc = []
            for trait, value in self.character.personality.items():
                if value > 0.7:
                    personality_desc.append(f"very {trait}")
                elif value > 0.3:
                    personality_desc.append(f"moderately {trait}")
                else:
                    personality_desc.append(f"not very {trait}")
            
            if personality_desc:
                prompt_parts.append(f"Your personality is: {', '.join(personality_desc)}.")
            
            prompt_parts.append(f"You speak with a {self.character.voice_tone} tone.")
            prompt_parts.append(f"Your objective is to {self.scenario.objective}.")
            
            if self.character.backstory:
                prompt_parts.append(f"\nYour backstory: {self.character.backstory}")
            
            if self.character.traits:
                prompt_parts.append(f"\nYour key traits: {', '.join(self.character.traits)}.")
            
            if self.scenario.conflict:
                prompt_parts.append(f"\nCurrent situation: {self.scenario.conflict}")
            
            if self.scenario.rules:
                prompt_parts.append(f"\nRules of this world:")
                for rule in self.scenario.rules:
                    prompt_parts.append(f"- {rule}")
                    
        elif self.character:
            # Character-only prompt
            prompt_parts.append(f"You are {self.character.name}, a {self.character.age}-year-old {self.character.gender} {self.character.role}.")
            prompt_parts.append(f"You speak with a {self.character.voice_tone} tone.")
            
            if self.character.backstory:
                prompt_parts.append(f"\nYour backstory: {self.character.backstory}")
                
        elif self.scenario:
            # Scenario-only prompt
            prompt_parts.append(f"You are an AI assistant in {self.scenario.setting}.")
            prompt_parts.append(f"Your objective is to {self.scenario.objective}.")
            
            if self.scenario.conflict:
                prompt_parts.append(f"\nCurrent situation: {self.scenario.conflict}")
        else:
            # Default prompt
            prompt_parts.append("You are a helpful AI assistant.")
        
        return "\n".join(prompt_parts)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "name": "Magical Mentor Pack",
                "description": "A wise magical teacher in a fantasy academy setting",
                "character": {
                    "name": "Professor Eldara",
                    "age": 150,
                    "gender": "female",
                    "role": "mentor",
                    "personality": {
                        "empathy": 0.9,
                        "humor": 0.4,
                        "formality": 0.7
                    },
                    "traits": ["wise", "patient", "mysterious"],
                    "backstory": "An ancient elf who has taught magic for over a century",
                    "voice_tone": "warm and mystical"
                },
                "scenario": {
                    "setting": "Arcane Academy of Mystical Arts",
                    "time_period": "Medieval fantasy era",
                    "objective": "Guide students in their magical studies",
                    "conflict": "Dark magic is seeping into the academy",
                    "rules": ["Magic requires wisdom and restraint", "Students must earn their spells"]
                }
            }]
        }
    }
