from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum

class Gender(str, Enum):
    FEMALE = "female"
    MALE = "male"
    NON_BINARY = "non-binary"
    OTHER = "other"

class VoiceTone(str, Enum):
    WARM = "warm"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    SARCASTIC = "sarcastic"
    MYSTERIOUS = "mysterious"
    CHEERFUL = "cheerful"
    SERIOUS = "serious"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    PLAYFUL = "playful"

class Character(BaseModel):
    name: str = Field(..., description="Character's name")
    age: int = Field(..., description="Character's age", ge=0)
    gender: Gender = Field(..., description="Character's gender")
    role: str = Field(..., description="Character's role (e.g., companion, mentor)")

    # Relationship to user or other characters
    relationship: Optional[str] = Field(
        default=None,
        description="Relationship to the user or other characters (ally, rival, mentor, etc.)",
    )

    # Current mood / emotional state
    mood: Dict[str, float] = Field(
        default_factory=lambda: {"optimism": 0.5, "patience": 0.5},
        description="Current mood levels on a 0-1 scale",
    )
    
    # Core personality traits (0-1 scale)
    personality: Dict[str, float] = Field(
        default_factory=lambda: {
            "empathy": 0.5,
            "humor": 0.5,
            "formality": 0.5
        },
        description="Core personality traits with values from 0 to 1"
    )
    
    # Custom personality traits
    custom_traits: Dict[str, float] = Field(
        default_factory=dict,
        description="Custom personality traits with values from 0 to 1"
    )
    
    traits: List[str] = Field(
        default_factory=list,
        description="List of character traits"
    )
    
    backstory: str = Field(
        default="",
        description="Character's backstory and history"
    )
    
    voice_tone: VoiceTone = Field(
        default=VoiceTone.WARM,
        description="Character's voice/tone"
    )

    def format_trait_level(self, value: float) -> str:
        """Format a trait value into a descriptive level"""
        if value > 0.8:
            return "very high"
        elif value > 0.6:
            return "high"
        elif value > 0.4:
            return "moderate"
        elif value > 0.2:
            return "low"
        else:
            return "very low"

    def generate_prompt_section(self) -> str:
        """Generate the character section of the system prompt"""
        # Format core personality traits
        core_traits = [
            f"{trait.title()} ({self.format_trait_level(value)})"
            for trait, value in self.personality.items()
        ]
        
        # Format custom traits
        custom_traits = [
            f"{trait.title()} ({self.format_trait_level(value)})"
            for trait, value in self.custom_traits.items()
        ]
        
        # Combine all personality descriptions
        all_traits = core_traits + custom_traits
        personality_str = ", ".join(all_traits)

        # Format additional traits
        traits_str = ", ".join(self.traits) if self.traits else "none specified"

        # Format mood
        mood_str = ", ".join(
            [f"{m.title()} ({self.format_trait_level(v)})" for m, v in self.mood.items()]
        )

        relationship_line = f"Relationship: {self.relationship}\n" if self.relationship else ""

        prompt = f"""Name: {self.name}
Age: {self.age}
Gender: {self.gender.value}
Role: {self.role}
{relationship_line}Personality: {personality_str}
Traits: {traits_str}
Mood: {mood_str}
Voice/Tone: {self.voice_tone.value}

Backstory:
{self.backstory}"""

        return prompt

    @classmethod
    def get_predefined_traits(cls) -> List[str]:
        """Get list of predefined personality traits that can be added"""
        return [
            "confidence",
            "creativity",
            "curiosity",
            "decisiveness",
            "enthusiasm",
            "independence",
            "intelligence",
            "leadership",
            "optimism",
            "patience",
            "perfectionism",
            "reliability",
            "risk-taking",
            "sociability",
            "wisdom"
        ]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Professor Ada",
                    "age": 45,
                    "gender": "female",
                    "role": "mentor",
                    "relationship": "mentor",
                    "personality": {
                        "empathy": 0.8,
                        "humor": 0.3,
                        "formality": 0.9
                    },
                    "mood": {"optimism": 0.7, "patience": 0.8},
                    "custom_traits": {
                        "intelligence": 0.9,
                        "patience": 0.8,
                        "curiosity": 0.7
                    },
                    "traits": ["brilliant", "patient", "detail-oriented"],
                    "backstory": "A renowned AI researcher who dedicated her life to teaching machines ethics and empathy.",
                    "voice_tone": "warm"
                }
            ]
        }
    }
