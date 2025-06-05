from pydantic import BaseModel, Field
from typing import Dict, List

class Character(BaseModel):
    name: str = Field(..., description="Character's name")
    age: int = Field(..., description="Character's age", ge=0)
    gender: str = Field(..., description="Character's gender")
    role: str = Field(..., description="Character's role (e.g., companion, mentor)")
    personality: Dict[str, float] = Field(
        default_factory=lambda: {
            "empathy": 0.5,
            "humor": 0.5,
            "formality": 0.5
        },
        description="Personality traits with values from 0 to 1"
    )
    traits: List[str] = Field(
        default_factory=list,
        description="List of character traits"
    )
    backstory: str = Field(
        default="",
        description="Character's backstory and history"
    )
    voice_tone: str = Field(
        default="neutral",
        description="Character's voice/tone (e.g., warm, sarcastic)"
    )

    def generate_prompt_section(self) -> str:
        """Generate the character section of the system prompt"""
        personality_traits = [
            f"{trait} ({value:.1f}/1.0)"
            for trait, value in self.personality.items()
        ]
        
        traits_str = ", ".join(self.traits) if self.traits else "none specified"
        
        prompt = f"""Name: {self.name}
Age: {self.age}
Gender: {self.gender}
Role: {self.role}
Personality: {', '.join(personality_traits)}
Traits: {traits_str}
Voice/Tone: {self.voice_tone}

Backstory:
{self.backstory}"""

        return prompt

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "name": "Professor Ada",
                "age": 45,
                "gender": "female",
                "role": "mentor",
                "personality": {
                    "empathy": 0.8,
                    "humor": 0.3,
                    "formality": 0.9
                },
                "traits": ["brilliant", "patient", "detail-oriented"],
                "backstory": "A renowned AI researcher who dedicated her life to teaching machines ethics and empathy.",
                "voice_tone": "warm and professional"
            }]
        }
    }
