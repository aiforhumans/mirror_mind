from typing import Optional, Dict, Any
from jinja2 import Environment, BaseLoader, Template
from models.character import Character
from models.scenario import Scenario

class StringTemplateLoader(BaseLoader):
    """Custom Jinja2 loader for string templates"""
    
    def __init__(self, templates: Dict[str, str]):
        self.templates = templates
    
    def get_source(self, environment: Environment, template: str) -> tuple:
        if template not in self.templates:
            raise FileNotFoundError(f"Template '{template}' not found")
        
        source = self.templates[template]
        return source, None, lambda: True

class PromptGenerator:
    """Enhanced prompt generator with Jinja2 templating support"""
    
    # Template definitions
    TEMPLATES = {
        "character_only": """You are {{ character.name }}, a {{ character.age }}-year-old {{ character.gender.value }} {{ character.role }}.

{% if character.personality or character.custom_traits %}
Your personality is characterized by:
{% for trait, value in character.personality.items() %}
- {{ trait.title() }}: {{ format_trait_level(value) }}
{% endfor %}
{% for trait, value in character.custom_traits.items() %}
- {{ trait.title() }}: {{ format_trait_level(value) }}
{% endfor %}
{% endif %}

You speak with a {{ character.voice_tone.value }} tone.

{% if character.traits %}
Your key traits include: {{ character.traits | join(', ') }}.
{% endif %}

{% if character.backstory %}
Your backstory: {{ character.backstory }}
{% endif %}""",

        "scenario_only": """You are an AI assistant operating in {{ scenario.setting }} during {{ scenario.time_period }}.

Your primary objective is to {{ scenario.objective }}.

{% if scenario.conflict %}
Current situation: {{ scenario.conflict }}
{% endif %}

{% if scenario.rules %}
Important rules and guidelines:
{% for rule in scenario.rules %}
- {{ rule }}
{% endfor %}
{% endif %}""",

        "character_and_scenario": """You are {{ character.name }}, a {{ character.age }}-year-old {{ character.gender.value }} {{ character.role }} in {{ scenario.setting }} during {{ scenario.time_period }}.

{% if character.personality or character.custom_traits %}
Your personality is characterized by:
{% for trait, value in character.personality.items() %}
- {{ trait.title() }}: {{ format_trait_level(value) }}
{% endfor %}
{% for trait, value in character.custom_traits.items() %}
- {{ trait.title() }}: {{ format_trait_level(value) }}
{% endfor %}
{% endif %}

You speak with a {{ character.voice_tone.value }} tone.

{% if character.traits %}
Your key traits include: {{ character.traits | join(', ') }}.
{% endif %}

Your objective is to {{ scenario.objective }}.

{% if character.backstory %}
Your backstory: {{ character.backstory }}
{% endif %}

{% if scenario.conflict %}
Current situation: {{ scenario.conflict }}
{% endif %}

{% if scenario.rules %}
Rules of this world:
{% for rule in scenario.rules %}
- {{ rule }}
{% endfor %}
{% endif %}""",

        "default": """You are a helpful AI assistant."""
    }
    
    def __init__(self):
        """Initialize the prompt generator with Jinja2 environment"""
        loader = StringTemplateLoader(self.TEMPLATES)
        self.env = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
        
        # Add custom functions to template environment
        self.env.globals['format_trait_level'] = self._format_trait_level
    
    def _format_trait_level(self, value: float) -> str:
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
    
    @staticmethod
    def generate_character_prompt(character: Character) -> str:
        """Generate a system prompt for a character (legacy method)"""
        return character.generate_prompt_section()
    
    @staticmethod
    def generate_scenario_prompt(scenario: Scenario) -> str:
        """Generate a system prompt for a scenario (legacy method)"""
        return scenario.generate_prompt_section()
    
    def generate_templated_prompt(
        self,
        character: Optional[Character] = None,
        scenario: Optional[Scenario] = None,
        template_name: Optional[str] = None,
        additional_instructions: str = ""
    ) -> str:
        """Generate a prompt using Jinja2 templates"""
        
        # Determine which template to use
        if template_name:
            if template_name not in self.TEMPLATES:
                raise ValueError(f"Template '{template_name}' not found")
            template_key = template_name
        else:
            # Auto-select template based on available data
            if character and scenario:
                template_key = "character_and_scenario"
            elif character:
                template_key = "character_only"
            elif scenario:
                template_key = "scenario_only"
            else:
                template_key = "default"
        
        # Load and render template
        template = self.env.get_template(template_key)
        
        # Prepare template context
        context = {
            'character': character,
            'scenario': scenario
        }
        
        # Render the main prompt
        prompt = template.render(**context)
        
        # Add additional instructions if provided
        if additional_instructions:
            prompt += f"\n\n=== Additional Instructions ===\n{additional_instructions}"
        
        return prompt.strip()
    
    @staticmethod
    def combine_prompts(
        character: Optional[Character] = None,
        scenario: Optional[Scenario] = None,
        additional_instructions: str = ""
    ) -> str:
        """Combine character and scenario prompts with optional instructions (legacy method)"""
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
        """Format personality traits into readable text (legacy method)"""
        descriptions = []
        for trait, value in traits.items():
            if value > 0.7:
                descriptions.append(f"very {trait}")
            elif value > 0.3:
                descriptions.append(f"moderately {trait}")
            else:
                descriptions.append(f"not very {trait}")
        return ", ".join(descriptions)
    
    def get_available_templates(self) -> list:
        """Get list of available template names"""
        return list(self.TEMPLATES.keys())
    
    def add_custom_template(self, name: str, template_string: str) -> None:
        """Add a custom template"""
        self.TEMPLATES[name] = template_string
        # Recreate the environment with updated templates
        loader = StringTemplateLoader(self.TEMPLATES)
        self.env = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
        self.env.globals['format_trait_level'] = self._format_trait_level
