from pydantic import BaseModel, Field
from typing import List

class Scenario(BaseModel):
    setting: str = Field(..., description="The setting/environment (e.g., cyberpunk Tokyo)")
    time_period: str = Field(..., description="Time period (past, present, future, etc.)")
    objective: str = Field(..., description="Main objective or goal of the scenario")
    conflict: str = Field(
        default="",
        description="Main conflict or tension in the scenario"
    )
    rules: List[str] = Field(
        default_factory=list,
        description="Rules or limitations for the scenario"
    )

    def generate_prompt_section(self) -> str:
        """Generate the scenario section of the system prompt"""
        rules_str = "\n".join([f"- {rule}" for rule in self.rules]) if self.rules else "- No specific rules"
        
        prompt = f"""Setting: {self.setting}
Time Period: {self.time_period}
Objective: {self.objective}
Conflict/Tension: {self.conflict if self.conflict else "None specified"}

World Rules:
{rules_str}"""

        return prompt

    def add_rule(self, rule: str) -> None:
        """Add a new rule to the scenario"""
        if rule.strip() and rule not in self.rules:
            self.rules.append(rule.strip())

    def remove_rule(self, index: int) -> None:
        """Remove a rule by index"""
        if 0 <= index < len(self.rules):
            self.rules.pop(index)

    def update_rule(self, index: int, new_rule: str) -> None:
        """Update a rule at the given index"""
        if 0 <= index < len(self.rules) and new_rule.strip():
            self.rules[index] = new_rule.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "setting": "A mystical academy for magical arts",
                    "time_period": "Medieval fantasy era",
                    "objective": "Guide students through their magical education",
                    "conflict": "Dark forces threaten the academy's ancient protective barriers",
                    "rules": [
                        "Magic has consequences and requires careful study",
                        "Students must respect the ancient traditions",
                        "No magic should be used to harm others"
                    ]
                }
            ]
        }
    }
