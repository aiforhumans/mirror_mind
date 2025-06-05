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
    auto_optimize: bool = Field(
        default=False,
        description="Whether to automatically optimize prompts when saving"
    )
    template_name: Optional[str] = Field(
        default=None,
        description="Custom template to use for prompt generation"
    )
    optimized_prompt: Optional[str] = Field(
        default=None,
        description="AI-optimized version of the system prompt"
    )

    def generate_system_prompt(self, use_templating: bool = True) -> str:
        """Generate the complete system prompt from character and scenario"""
        if use_templating:
            # Use the new templating system
            from utils.prompt_generator import PromptGenerator
            generator = PromptGenerator()
            return generator.generate_templated_prompt(
                character=self.character,
                scenario=self.scenario,
                template_name=self.template_name
            )
        else:
            # Use legacy string concatenation method
            return self._generate_legacy_prompt()

    def _generate_legacy_prompt(self) -> str:
        """Generate prompt using the original string concatenation method"""
        prompt_parts = []
        
        if self.character and self.scenario:
            # Full character + scenario prompt
            prompt_parts.append(f"You are {self.character.name}, a {self.character.age}-year-old {self.character.gender.value} {self.character.role} in {self.scenario.setting} during {self.scenario.time_period}.")
            
            # Add personality description
            personality_desc = []
            for trait, value in self.character.personality.items():
                if value > 0.7:
                    personality_desc.append(f"very {trait}")
                elif value > 0.3:
                    personality_desc.append(f"moderately {trait}")
                else:
                    personality_desc.append(f"not very {trait}")
            
            # Add custom traits
            for trait, value in self.character.custom_traits.items():
                if value > 0.7:
                    personality_desc.append(f"very {trait}")
                elif value > 0.3:
                    personality_desc.append(f"moderately {trait}")
                else:
                    personality_desc.append(f"not very {trait}")
            
            if personality_desc:
                prompt_parts.append(f"Your personality is: {', '.join(personality_desc)}.")
            
            prompt_parts.append(f"You speak with a {self.character.voice_tone.value} tone.")
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
            prompt_parts.append(f"You are {self.character.name}, a {self.character.age}-year-old {self.character.gender.value} {self.character.role}.")
            prompt_parts.append(f"You speak with a {self.character.voice_tone.value} tone.")
            
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

    def get_preview_prompt(self, use_templating: bool = True) -> str:
        """Get a preview of the system prompt with highlighting for missing sections"""
        prompt = self.generate_system_prompt(use_templating=use_templating)
        
        # Add warnings for missing sections
        warnings = []
        if not self.character and not self.scenario:
            warnings.append("⚠️ No character or scenario selected")
        elif not self.character:
            warnings.append("ℹ️ No character selected - using generic AI assistant")
        elif not self.scenario:
            warnings.append("ℹ️ No scenario selected - character without world context")
        
        if self.character:
            if not self.character.backstory:
                warnings.append("💡 Consider adding a backstory for richer character depth")
            if not self.character.traits:
                warnings.append("💡 Consider adding character traits")
        
        if self.scenario:
            if not self.scenario.conflict:
                warnings.append("💡 Consider adding conflict/tension for more engaging scenarios")
            if not self.scenario.rules:
                warnings.append("💡 Consider adding world rules for consistency")
        
        if warnings:
            warning_text = "\n".join(warnings)
            return f"{prompt}\n\n=== Preview Notes ===\n{warning_text}"
        
        return prompt

    def validate_completeness(self) -> dict:
        """Validate the completeness of the prompt pack and return suggestions"""
        issues = []
        suggestions = []
        
        # Check for basic requirements
        if not self.character and not self.scenario:
            issues.append("No character or scenario defined")
        
        # Character validation
        if self.character:
            if not self.character.backstory:
                suggestions.append("Add character backstory for depth")
            if not self.character.traits:
                suggestions.append("Add character traits for personality")
            if len(self.character.personality) == 3 and not self.character.custom_traits:
                suggestions.append("Consider adding custom personality traits")
        
        # Scenario validation
        if self.scenario:
            if not self.scenario.conflict:
                suggestions.append("Add conflict/tension for engagement")
            if not self.scenario.rules:
                suggestions.append("Add world rules for consistency")
        
        return {
            "is_complete": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "completeness_score": self._calculate_completeness_score()
        }

    def _calculate_completeness_score(self) -> float:
        """Calculate a completeness score from 0.0 to 1.0"""
        score = 0.0
        max_score = 0.0
        
        # Base score for having character or scenario
        if self.character or self.scenario:
            score += 0.3
        max_score += 0.3
        
        # Character scoring
        if self.character:
            max_score += 0.4
            score += 0.1  # Base character score
            
            if self.character.backstory:
                score += 0.1
            if self.character.traits:
                score += 0.1
            if self.character.custom_traits:
                score += 0.1
        
        # Scenario scoring
        if self.scenario:
            max_score += 0.3
            score += 0.1  # Base scenario score
            
            if self.scenario.conflict:
                score += 0.1
            if self.scenario.rules:
                score += 0.1
        
        return score / max_score if max_score > 0 else 0.0

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Magical Mentor Pack",
                    "description": "A wise magical teacher in a fantasy academy setting",
                    "auto_optimize": True,
                    "template_name": "character_and_scenario",
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
                        "custom_traits": {
                            "wisdom": 0.9,
                            "patience": 0.8
                        },
                        "traits": ["wise", "patient", "mysterious"],
                        "backstory": "An ancient elf who has taught magic for over a century",
                        "voice_tone": "warm"
                    },
                    "scenario": {
                        "setting": "Arcane Academy of Mystical Arts",
                        "time_period": "Medieval fantasy era",
                        "objective": "Guide students in their magical studies",
                        "conflict": "Dark magic is seeping into the academy",
                        "rules": ["Magic requires wisdom and restraint", "Students must earn their spells"]
                    }
                }
            ]
        }
    }
