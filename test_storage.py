#!/usr/bin/env python3
"""Test script to debug storage functionality"""

from utils.storage import Storage
from models.character import Character
from models.scenario import Scenario

def test_character_storage():
    print("Testing character storage...")
    storage = Storage()
    
    # Create a test character
    character = Character(
        name="Test Character",
        age=25,
        gender="non-binary",
        role="companion",
        personality={"empathy": 0.8, "humor": 0.6, "formality": 0.3},
        traits=["friendly", "curious"],
        backstory="A test character for debugging",
        voice_tone="casual"
    )
    
    try:
        # Test saving
        print("Attempting to save character...")
        filepath = storage.save_character(character)
        print(f"Character saved to: {filepath}")
        
        # Test loading
        print("Attempting to load character...")
        loaded_character = storage.load_character("test_character.json")
        if loaded_character:
            print(f"Character loaded successfully: {loaded_character.name}")
            print(f"Character data: {loaded_character.model_dump()}")
        else:
            print("Failed to load character")
            
        # Test listing
        print("Listing characters...")
        character_names = storage.get_character_names()
        print(f"Found characters: {character_names}")
        
    except Exception as e:
        print(f"Error during character storage test: {e}")
        import traceback
        traceback.print_exc()

def test_scenario_storage():
    print("\nTesting scenario storage...")
    storage = Storage()
    
    # Create a test scenario
    scenario = Scenario(
        setting="Test Environment",
        time_period="present day",
        objective="test the storage system",
        conflict="debugging issues",
        rules=["be thorough", "check for errors"]
    )
    
    try:
        # Test saving
        print("Attempting to save scenario...")
        filepath = storage.save_scenario(scenario)
        print(f"Scenario saved to: {filepath}")
        
        # Test loading
        print("Attempting to load scenario...")
        loaded_scenario = storage.load_scenario("test_environment.json")
        if loaded_scenario:
            print(f"Scenario loaded successfully: {loaded_scenario.setting}")
            print(f"Scenario data: {loaded_scenario.model_dump()}")
        else:
            print("Failed to load scenario")
            
        # Test listing
        print("Listing scenarios...")
        scenario_names = storage.get_scenario_names()
        print(f"Found scenarios: {scenario_names}")
        
    except Exception as e:
        print(f"Error during scenario storage test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_character_storage()
    test_scenario_storage()
