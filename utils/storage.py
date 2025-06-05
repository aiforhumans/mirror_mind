import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from models.character import Character
from models.scenario import Scenario
from models.prompt_pack import PromptPack

class Storage:
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.characters_path = self.base_path / "characters"
        self.scenarios_path = self.base_path / "scenarios"
        self.prompt_packs_path = self.base_path / "prompt_packs"
        
        # Create directories if they don't exist
        self.characters_path.mkdir(parents=True, exist_ok=True)
        self.scenarios_path.mkdir(parents=True, exist_ok=True)
        self.prompt_packs_path.mkdir(parents=True, exist_ok=True)
    
    def save_character(self, character: Character, filename: Optional[str] = None) -> str:
        """Save a character to JSON file"""
        if filename is None:
            filename = f"{character.name.lower().replace(' ', '_')}.json"
        
        filepath = self.characters_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(character.model_dump(), f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def load_character(self, filename: str) -> Optional[Character]:
        """Load a character from JSON file"""
        filepath = self.characters_path / filename
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Character(**data)
        except Exception as e:
            print(f"Error loading character {filename}: {e}")
            return None
    
    def list_characters(self) -> List[str]:
        """List all saved character files"""
        return [f.name for f in self.characters_path.glob("*.json")]
    
    def delete_character(self, filename: str) -> bool:
        """Delete a character file"""
        filepath = self.characters_path / filename
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    def save_scenario(self, scenario: Scenario, filename: Optional[str] = None) -> str:
        """Save a scenario to JSON file"""
        if filename is None:
            # Create filename from setting
            safe_name = scenario.setting.lower().replace(' ', '_')[:30]
            filename = f"{safe_name}.json"
        
        filepath = self.scenarios_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(scenario.model_dump(), f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def load_scenario(self, filename: str) -> Optional[Scenario]:
        """Load a scenario from JSON file"""
        filepath = self.scenarios_path / filename
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Scenario(**data)
        except Exception as e:
            print(f"Error loading scenario {filename}: {e}")
            return None
    
    def list_scenarios(self) -> List[str]:
        """List all saved scenario files"""
        return [f.name for f in self.scenarios_path.glob("*.json")]
    
    def delete_scenario(self, filename: str) -> bool:
        """Delete a scenario file"""
        filepath = self.scenarios_path / filename
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    def save_prompt_pack(self, prompt_pack: PromptPack, filename: Optional[str] = None) -> str:
        """Save a prompt pack to JSON file"""
        if filename is None:
            filename = f"{prompt_pack.name.lower().replace(' ', '_')}.json"
        
        filepath = self.prompt_packs_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prompt_pack.model_dump(), f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def load_prompt_pack(self, filename: str) -> Optional[PromptPack]:
        """Load a prompt pack from JSON file"""
        filepath = self.prompt_packs_path / filename
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return PromptPack(**data)
        except Exception as e:
            print(f"Error loading prompt pack {filename}: {e}")
            return None
    
    def list_prompt_packs(self) -> List[str]:
        """List all saved prompt pack files"""
        return [f.name for f in self.prompt_packs_path.glob("*.json")]
    
    def delete_prompt_pack(self, filename: str) -> bool:
        """Delete a prompt pack file"""
        filepath = self.prompt_packs_path / filename
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    def get_character_names(self) -> List[str]:
        """Get list of character names (without .json extension)"""
        return [f.stem for f in self.characters_path.glob("*.json")]
    
    def get_scenario_names(self) -> List[str]:
        """Get list of scenario names (without .json extension)"""
        return [f.stem for f in self.scenarios_path.glob("*.json")]
    
    def get_prompt_pack_names(self) -> List[str]:
        """Get list of prompt pack names (without .json extension)"""
        return [f.stem for f in self.prompt_packs_path.glob("*.json")]
