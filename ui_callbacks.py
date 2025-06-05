"""Separate callback functions used by the Gradio UI."""

from typing import List, Dict
import gradio as gr

from models.character import Character, Gender, VoiceTone
from models.scenario import Scenario
from models.prompt_pack import PromptPack


def handle_send(chat_bot, message, history, system_message, model, temp, max_tok, top_p_val, freq_pen, pres_pen, use_stream):
    """Send a message to the model and optionally stream the response."""
    if not message.strip():
        return history, ""

    if use_stream:
        for new_history, _ in chat_bot.stream_chat_with_lm_studio(
            message,
            history,
            system_message,
            model,
            temp,
            max_tok,
            top_p_val,
            freq_pen,
            pres_pen,
        ):
            yield new_history, ""
    else:
        result = chat_bot.chat_with_lm_studio(
            message,
            history,
            system_message,
            model,
            temp,
            max_tok,
            top_p_val,
            freq_pen,
            pres_pen,
        )
        return result


def refresh_models(chat_bot):
    new_models = chat_bot.get_available_models()
    return gr.Dropdown(choices=new_models, value=new_models[0] if new_models else "default-model")


def refresh_prompt_packs(storage):
    packs = ["None"] + storage.get_prompt_pack_names()
    return gr.Dropdown(choices=packs, value="None")


def add_custom_trait(trait_name, trait_value, current_traits: Dict[str, float]):
    if trait_name and trait_name not in current_traits:
        current_traits[trait_name] = trait_value
        display_text = "\n".join([f"{name}: {value:.1f}" for name, value in current_traits.items()])
        return current_traits, display_text
    return current_traits, gr.update()


def update_character_preview(name, age, gender, role, relationship, empathy, humor, formality, optimism, patience, traits, voice_tone, backstory, custom_traits):
    if not name:
        return "", ""

    try:
        traits_list = [t.strip() for t in traits.split(",") if t.strip()] if traits else []
        character = Character(
            name=name,
            age=int(age) if age else 30,
            gender=Gender(gender),
            role=role,
            relationship=relationship or None,
            personality={"empathy": empathy, "humor": humor, "formality": formality},
            mood={"optimism": optimism, "patience": patience},
            custom_traits=custom_traits,
            traits=traits_list,
            backstory=backstory or "",
            voice_tone=VoiceTone(voice_tone),
        )

        preview = character.generate_prompt_section()

        completeness_items = []
        if character.backstory:
            completeness_items.append("✅ Backstory")
        else:
            completeness_items.append("❌ Backstory missing")

        if character.traits:
            completeness_items.append("✅ Traits defined")
        else:
            completeness_items.append("❌ No traits defined")

        if character.custom_traits:
            completeness_items.append("✅ Custom personality traits")
        else:
            completeness_items.append("💡 Consider adding custom traits")

        completeness = "\n".join(completeness_items)
        return preview, completeness
    except Exception as e:
        return f"Error: {str(e)}", "❌ Invalid character data"


def add_scenario_rule(new_rule: str, current_rules: List[str]):
    if new_rule.strip() and new_rule not in current_rules:
        current_rules.append(new_rule.strip())
        display_text = "\n".join([f"{i}: {rule}" for i, rule in enumerate(current_rules)])
        return current_rules, display_text, ""
    return current_rules, gr.update(), new_rule


def edit_scenario_rule(rule_idx: int, new_text: str, current_rules: List[str]):
    if 0 <= rule_idx < len(current_rules) and new_text.strip():
        current_rules[rule_idx] = new_text.strip()
        display_text = "\n".join([f"{i}: {rule}" for i, rule in enumerate(current_rules)])
        return current_rules, display_text
    return current_rules, gr.update()


def delete_scenario_rule(rule_idx: int, current_rules: List[str]):
    if 0 <= rule_idx < len(current_rules):
        current_rules.pop(rule_idx)
        display_text = "\n".join([f"{i}: {rule}" for i, rule in enumerate(current_rules)])
        return current_rules, display_text
    return current_rules, gr.update()


def update_scenario_preview(setting, time_period, objective, conflict, env_tone, culture, hooks, rules_list: List[str]):
    if not setting:
        return "", ""

    try:
        scenario = Scenario(
            setting=setting,
            time_period=time_period or "present",
            objective=objective or "assist the user",
            conflict=conflict or "",
            environmental_tone=env_tone or "",
            cultural_influences=culture or "",
            story_hooks=hooks or "",
            rules=rules_list,
        )

        preview = scenario.generate_prompt_section()

        completeness_items = []
        if scenario.conflict:
            completeness_items.append("✅ Conflict/tension defined")
        else:
            completeness_items.append("💡 Consider adding conflict")

        if scenario.rules:
            completeness_items.append(f"✅ {len(scenario.rules)} rules defined")
        else:
            completeness_items.append("💡 Consider adding rules")

        completeness = "\n".join(completeness_items)
        return preview, completeness
    except Exception as e:
        return f"Error: {str(e)}", "❌ Invalid scenario data"


def update_pack_preview_and_validation(storage, char_name, char2_name, scenario_name, template_name, interaction):
    character = None
    scenario = None

    if char_name and char_name != "None":
        character = storage.load_character(f"{char_name}.json")
    if char2_name and char2_name != "None":
        character2 = storage.load_character(f"{char2_name}.json")
    else:
        character2 = None

    if scenario_name and scenario_name != "None":
        scenario = storage.load_scenario(f"{scenario_name}.json")

    if character or character2 or scenario:
        pack = PromptPack(
            name="Preview",
            character=character,
            secondary_character=character2,
            character_interaction=interaction or None,
            scenario=scenario,
            template_name=template_name,
        )

        preview = pack.get_preview_prompt(use_templating=True)
        validation = pack.validate_completeness()

        validation_text = f"Completeness Score: {validation['completeness_score']:.1%}\n"
        if validation['issues']:
            validation_text += "Issues: " + ", ".join(validation['issues']) + "\n"
        if validation['suggestions']:
            validation_text += "Suggestions: " + ", ".join(validation['suggestions'])

        return preview, validation_text

    return "No character or scenario selected", "Select a character and/or scenario to see preview"


def save_character(storage, name, age, gender, role, relationship, empathy, humor, formality, optimism, patience, traits, voice_tone, backstory, custom_traits):
    if not name:
        return "Error: Character name is required", gr.Dropdown(), gr.Dropdown()

    try:
        traits_list = [t.strip() for t in traits.split(",") if t.strip()] if traits else []
        character = Character(
            name=name,
            age=int(age) if age else 30,
            gender=Gender(gender),
            role=role,
            relationship=relationship or None,
            personality={"empathy": empathy, "humor": humor, "formality": formality},
            mood={"optimism": optimism, "patience": patience},
            custom_traits=custom_traits,
            traits=traits_list,
            backstory=backstory or "",
            voice_tone=VoiceTone(voice_tone),
        )
        storage.save_character(character)
        char_names = storage.get_character_names()
        return (
            f"Character '{name}' saved successfully!",
            gr.Dropdown(choices=char_names),
            gr.Dropdown(choices=["None"] + char_names),
        )
    except Exception as e:
        return f"Error saving character: {str(e)}", gr.Dropdown(), gr.Dropdown()


def load_character(storage, char_name):
    if not char_name:
        return [
            "",
            "",
            "",
            "",
            "",
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            "",
            "",
            "",
            {},
            "",
        ]

    character = storage.load_character(f"{char_name}.json")
    if character:
        traits_str = ", ".join(character.traits)
        custom_traits_display = "\n".join([f"{name}: {value:.1f}" for name, value in character.custom_traits.items()])
        return [
            character.name,
            character.age,
            character.gender.value,
            character.role,
            character.relationship or "",
            character.personality.get("empathy", 0.5),
            character.personality.get("humor", 0.5),
            character.personality.get("formality", 0.5),
            character.mood.get("optimism", 0.5),
            character.mood.get("patience", 0.5),
            traits_str,
            character.voice_tone.value,
            character.backstory,
            character.custom_traits,
            custom_traits_display,
        ]
    return [
        "",
        "",
        "",
        "",
        "",
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        "",
        "",
        "",
        {},
        "",
    ]


def save_scenario(storage, setting, time_period, objective, conflict, env_tone, culture, hooks, rules_list):
    if not setting:
        return "Error: Setting is required", gr.Dropdown(), gr.Dropdown()

    try:
        scenario = Scenario(
            setting=setting,
            time_period=time_period or "present",
            objective=objective or "assist the user",
            conflict=conflict or "",
            environmental_tone=env_tone or "",
            cultural_influences=culture or "",
            story_hooks=hooks or "",
            rules=rules_list,
        )
        storage.save_scenario(scenario)
        scenario_names = storage.get_scenario_names()
        return (
            "Scenario saved successfully!",
            gr.Dropdown(choices=scenario_names),
            gr.Dropdown(choices=["None"] + scenario_names),
        )
    except Exception as e:
        return f"Error saving scenario: {str(e)}", gr.Dropdown(), gr.Dropdown()


def load_scenario(storage, scenario_name):
    if not scenario_name:
        return ["" ] * 7 + [[]] + [""]

    scenario = storage.load_scenario(f"{scenario_name}.json")
    if scenario:
        rules_display = "\n".join([f"{i}: {rule}" for i, rule in enumerate(scenario.rules)])
        return [
            scenario.setting,
            scenario.time_period,
            scenario.objective,
            scenario.conflict,
            scenario.environmental_tone,
            scenario.cultural_influences,
            scenario.story_hooks,
            scenario.rules,
            rules_display,
        ]
    return ["" ] * 7 + [[]] + [""]


def optimize_prompt_with_ai(chat_bot, prompt_text: str, model: str):
    if not prompt_text:
        return "", gr.update(visible=False), gr.update(visible=False)

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at rewriting and optimizing system prompts for AI language models. "
                    "Your task is to rewrite the following prompt to improve its natural tone, engagement, and effectiveness, "
                    "while preserving all core instructions, character traits, and behavioral guidelines. "
                    "The result must start with **'You are ...'**, phrased as a system prompt, not a message to the user. "
                    "Do not add any explanation or commentary—only output the improved system prompt as plain text. "
                    "include rules and guidelines in a clear, concise manner."
                ),
            },
            {"role": "user", "content": f"Rewrite and enhance the following system prompt:\n\n{prompt_text}"},
        ]

        completion = chat_bot.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=False,
        )

        optimized = completion.choices[0].message.content
        return optimized, gr.update(visible=True), gr.update(visible=True)
    except Exception as e:
        return f"Error optimizing prompt: {str(e)}", gr.update(visible=False), gr.update(visible=False)


def save_prompt_pack(storage, chat_bot, name, description, char_name, char2_name, scenario_name, template_name, interaction, auto_optimize, model):
    if not name:
        return "Error: Pack name is required", gr.Dropdown()

    try:
        character = None
        character2 = None
        scenario = None

        if char_name and char_name != "None":
            character = storage.load_character(f"{char_name}.json")
        if char2_name and char2_name != "None":
            character2 = storage.load_character(f"{char2_name}.json")

        if scenario_name and scenario_name != "None":
            scenario = storage.load_scenario(f"{scenario_name}.json")

        pack = PromptPack(
            name=name,
            description=description or "",
            character=character,
            secondary_character=character2,
            character_interaction=interaction or None,
            scenario=scenario,
            template_name=template_name,
            auto_optimize=auto_optimize,
        )

        if auto_optimize and (character or character2 or scenario):
            try:
                base_prompt = pack.generate_system_prompt(use_templating=True)
                optimized, _, _ = optimize_prompt_with_ai(chat_bot, base_prompt, model)
                if optimized and "Error" not in optimized:
                    pack.optimized_prompt = optimized
            except Exception as e:
                print(f"Auto-optimization failed: {e}")

        storage.save_prompt_pack(pack)
        return f"Prompt pack '{name}' saved successfully!", gr.Dropdown(choices=storage.get_prompt_pack_names())
    except Exception as e:
        return f"Error saving prompt pack: {str(e)}", gr.Dropdown()


def load_prompt_pack_to_chat(storage, pack_name):
    if not pack_name or pack_name == "None":
        return "You are a helpful, harmless, and honest assistant."

    pack = storage.load_prompt_pack(f"{pack_name}.json")
    if pack:
        if pack.optimized_prompt:
            return pack.optimized_prompt
        return pack.generate_system_prompt(use_templating=True)
    return "You are a helpful, harmless, and honest assistant."


def load_prompt_pack_ui(storage, pack_name):
    if not pack_name or pack_name == "None":
        return (
            "",
            "",
            "None",
            "None",
            "None",
            "character_and_scenario",
            "",
            False,
            "",
            "",
        )

    pack = storage.load_prompt_pack(f"{pack_name}.json")
    if not pack:
        return (
            "",
            "",
            "None",
            "None",
            "None",
            "character_and_scenario",
            "",
            False,
            "",
            "",
        )

    preview = pack.get_preview_prompt(use_templating=True)
    validation = pack.validate_completeness()
    validation_text = f"Completeness Score: {validation['completeness_score']:.1%}\n"
    if validation['issues']:
        validation_text += "Issues: " + ", ".join(validation['issues']) + "\n"
    if validation['suggestions']:
        validation_text += "Suggestions: " + ", ".join(validation['suggestions'])

    return (
        pack.name,
        pack.description or "",
        pack.character.name if pack.character else "None",
        pack.secondary_character.name if pack.secondary_character else "None",
        pack.scenario.setting if pack.scenario else "None",
        pack.template_name or "character_and_scenario",
        pack.character_interaction or "",
        pack.auto_optimize,
        preview,
        validation_text,
    )

