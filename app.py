import gradio as gr
import openai
from openai import OpenAI
import json
import asyncio
from typing import List, Tuple, Optional

from utils.storage import Storage
from utils.prompt_generator import PromptGenerator
from models.character import Character
from models.scenario import Scenario
from models.prompt_pack import PromptPack

class LMStudioChat:
    def __init__(self):
        # Initialize OpenAI client pointing to LM Studio
        self.client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio"  # LM Studio doesn't require a real API key
        )
        self.conversation_history = []
        
    def get_available_models(self):
        """Get list of available models from LM Studio"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return ["default-model"]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return [], ""
    
    def chat_with_lm_studio(
        self, 
        message: str, 
        history: List[Tuple[str, str]], 
        system_message: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float
    ):
        """Send message to LM Studio and get response"""
        try:
            # Build messages array for the API
            messages = []
            
            # Add system message if provided
            if system_message.strip():
                messages.append({"role": "system", "content": system_message})
            
            # Add conversation history
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            
            # Add current user message
            messages.append({"role": "user", "content": message})
            
            # Make API call to LM Studio
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=False
            )
            
            # Extract response
            response = completion.choices[0].message.content
            
            # Update history
            history.append((message, response))
            
            return history, ""
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append((message, error_msg))
            return history, ""
    
    def stream_chat_with_lm_studio(
        self, 
        message: str, 
        history: List[Tuple[str, str]], 
        system_message: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float
    ):
        """Stream response from LM Studio"""
        try:
            # Build messages array for the API
            messages = []
            
            # Add system message if provided
            if system_message.strip():
                messages.append({"role": "system", "content": system_message})
            
            # Add conversation history
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            
            # Add current user message
            messages.append({"role": "user", "content": message})
            
            # Add user message to history immediately
            history.append((message, ""))
            
            # Make streaming API call to LM Studio
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=True
            )
            
            # Stream the response
            partial_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    partial_response += chunk.choices[0].delta.content
                    # Update the last message in history with partial response
                    history[-1] = (message, partial_response)
                    yield history, ""
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if history and history[-1][0] == message:
                history[-1] = (message, error_msg)
            else:
                history.append((message, error_msg))
            yield history, ""

def create_chat_interface():
    """Create the Gradio chat interface with persona and scenario features"""
    chat_bot = LMStudioChat()
    storage = Storage()
    
    # Get available models
    available_models = chat_bot.get_available_models()
    
    with gr.Blocks(title="LM Studio Chat with Personas", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 LM Studio Chat with AI Personas & Scenarios")
        gr.Markdown("Create AI characters and immersive scenarios for enhanced conversations")
        
        with gr.Tabs():
            # Chat Tab
            with gr.TabItem("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        # Chat interface
                        chatbot = gr.Chatbot(
                            height=500,
                            show_label=False,
                            container=True
                        )
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Type your message here...",
                                show_label=False,
                                scale=4,
                                container=False
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                        
                        with gr.Row():
                            clear_btn = gr.Button("Clear Chat", variant="secondary")
                            stream_toggle = gr.Checkbox(label="Stream Response", value=True)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Settings")
                        
                        # Model selection
                        model_dropdown = gr.Dropdown(
                            choices=available_models,
                            value=available_models[0] if available_models else "default-model",
                            label="Model",
                            info="Select the model to use"
                        )
                        
                        # Prompt Pack Selection
                        prompt_pack_dropdown = gr.Dropdown(
                            choices=["None"] + storage.get_prompt_pack_names(),
                            value="None",
                            label="Prompt Pack",
                            info="Select a saved prompt pack"
                        )
                        
                        # System message (will be auto-populated by prompt pack)
                        system_msg = gr.Textbox(
                            label="System Message",
                            placeholder="You are a helpful assistant...",
                            lines=5,
                            value="You are a helpful, harmless, and honest assistant."
                        )
                        
                        gr.Markdown("### 🎛️ Generation Parameters")
                        
                        # Temperature
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                            info="Controls randomness"
                        )
                        
                        # Max tokens
                        max_tokens = gr.Slider(
                            minimum=1,
                            maximum=4096,
                            value=1024,
                            step=1,
                            label="Max Tokens"
                        )
                        
                        # Top-p
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-p"
                        )
                        
                        # Frequency penalty
                        freq_penalty = gr.Slider(
                            minimum=-2.0,
                            maximum=2.0,
                            value=0.0,
                            step=0.1,
                            label="Frequency Penalty"
                        )
                        
                        # Presence penalty
                        presence_penalty = gr.Slider(
                            minimum=-2.0,
                            maximum=2.0,
                            value=0.0,
                            step=0.1,
                            label="Presence Penalty"
                        )
                        
                        # Refresh buttons
                        refresh_models_btn = gr.Button("🔄 Refresh Models", variant="secondary")
                        refresh_packs_btn = gr.Button("🔄 Refresh Packs", variant="secondary")
            
            # Character Creator Tab
            with gr.TabItem("👤 Character Creator"):
                gr.Markdown("### Create AI Character Personas")
                
                with gr.Row():
                    with gr.Column():
                        char_name = gr.Textbox(label="Name", placeholder="e.g., Professor Ada")
                        char_age = gr.Number(label="Age", value=30, minimum=0)
                        char_gender = gr.Textbox(label="Gender", placeholder="e.g., female, male, non-binary")
                        char_role = gr.Dropdown(
                            choices=["companion", "mentor", "adversary", "assistant", "teacher", "friend", "guide"],
                            label="Role",
                            value="companion"
                        )
                        
                        gr.Markdown("#### Personality Traits (0.0 - 1.0)")
                        char_empathy = gr.Slider(0.0, 1.0, 0.5, label="Empathy")
                        char_humor = gr.Slider(0.0, 1.0, 0.5, label="Humor")
                        char_formality = gr.Slider(0.0, 1.0, 0.5, label="Formality")
                        
                        char_traits = gr.Textbox(
                            label="Traits (comma-separated)",
                            placeholder="e.g., wise, patient, curious"
                        )
                        
                        char_voice_tone = gr.Dropdown(
                            choices=["warm", "professional", "casual", "sarcastic", "mysterious", "cheerful", "serious"],
                            label="Voice/Tone",
                            value="warm"
                        )
                    
                    with gr.Column():
                        char_backstory = gr.Textbox(
                            label="Backstory",
                            lines=8,
                            placeholder="Describe the character's background, history, and motivations..."
                        )
                        
                        # Character preview
                        char_preview = gr.Textbox(
                            label="Character Prompt Preview",
                            lines=10,
                            interactive=False
                        )
                        
                        # Save/Load controls
                        with gr.Row():
                            save_char_btn = gr.Button("💾 Save Character", variant="primary")
                            load_char_dropdown = gr.Dropdown(
                                choices=storage.get_character_names(),
                                label="Load Character"
                            )
                            load_char_btn = gr.Button("📂 Load")
                            delete_char_btn = gr.Button("🗑️ Delete", variant="stop")
            
            # Scenario Designer Tab
            with gr.TabItem("🌍 Scenario Designer"):
                gr.Markdown("### Create Immersive Scenarios")
                
                with gr.Row():
                    with gr.Column():
                        scenario_setting = gr.Textbox(
                            label="Setting",
                            placeholder="e.g., cyberpunk Tokyo, medieval village, space station"
                        )
                        scenario_time = gr.Textbox(
                            label="Time Period",
                            placeholder="e.g., future 2077, medieval era, present day"
                        )
                        scenario_objective = gr.Textbox(
                            label="Objective",
                            placeholder="e.g., solve a mystery, teach magic, explore new worlds"
                        )
                        scenario_conflict = gr.Textbox(
                            label="Conflict/Tension",
                            placeholder="e.g., ancient evil awakening, corporate conspiracy"
                        )
                        scenario_rules = gr.Textbox(
                            label="Rules (one per line)",
                            lines=5,
                            placeholder="e.g., Magic has consequences\nNo breaking character\nStay in medieval setting"
                        )
                    
                    with gr.Column():
                        # Scenario preview
                        scenario_preview = gr.Textbox(
                            label="Scenario Prompt Preview",
                            lines=10,
                            interactive=False
                        )
                        
                        # Save/Load controls
                        with gr.Row():
                            save_scenario_btn = gr.Button("💾 Save Scenario", variant="primary")
                            load_scenario_dropdown = gr.Dropdown(
                                choices=storage.get_scenario_names(),
                                label="Load Scenario"
                            )
                            load_scenario_btn = gr.Button("📂 Load")
                            delete_scenario_btn = gr.Button("🗑️ Delete", variant="stop")
            
            # Prompt Packs Tab
            with gr.TabItem("📦 Prompt Packs"):
                gr.Markdown("### Combine Characters & Scenarios")
                
                with gr.Row():
                    with gr.Column():
                        pack_name = gr.Textbox(label="Pack Name", placeholder="e.g., Magical Mentor")
                        pack_description = gr.Textbox(
                            label="Description",
                            lines=2,
                            placeholder="Brief description of this prompt pack"
                        )
                        
                        pack_character_dropdown = gr.Dropdown(
                            choices=["None"] + storage.get_character_names(),
                            label="Select Character",
                            value="None"
                        )
                        
                        pack_scenario_dropdown = gr.Dropdown(
                            choices=["None"] + storage.get_scenario_names(),
                            label="Select Scenario",
                            value="None"
                        )
                        
                        # Save/Load controls
                        with gr.Row():
                            save_pack_btn = gr.Button("💾 Save Pack", variant="primary")
                            load_pack_dropdown = gr.Dropdown(
                                choices=storage.get_prompt_pack_names(),
                                label="Load Pack"
                            )
                            load_pack_btn = gr.Button("📂 Load")
                            delete_pack_btn = gr.Button("🗑️ Delete", variant="stop")
                    
                    with gr.Column():
                        # Combined prompt preview
                        pack_preview = gr.Textbox(
                            label="Combined System Prompt Preview",
                            lines=15,
                            interactive=False
                        )
                        
                        # AI Optimization button
                        optimize_prompt_btn = gr.Button("🤖 AI Optimize Prompt", variant="secondary")
                        
                        # Optimized prompt display
                        optimized_prompt = gr.Textbox(
                            label="AI Optimized System Prompt",
                            lines=15,
                            interactive=True,
                            visible=False
                        )
        
        # Connection status
        with gr.Row():
            status = gr.Markdown("**Status:** Ready to chat! Make sure LM Studio is running on localhost:1234")
        
        # Event handlers
        def handle_send(message, history, system_message, model, temp, max_tok, top_p_val, freq_pen, pres_pen, use_stream):
            if not message.strip():
                return history, ""
            
            if use_stream:
                for new_history, _ in chat_bot.stream_chat_with_lm_studio(
                    message, history, system_message, model, temp, max_tok, top_p_val, freq_pen, pres_pen
                ):
                    yield new_history, ""
            else:
                result = chat_bot.chat_with_lm_studio(
                    message, history, system_message, model, temp, max_tok, top_p_val, freq_pen, pres_pen
                )
                return result
        
        def refresh_models():
            new_models = chat_bot.get_available_models()
            return gr.Dropdown(choices=new_models, value=new_models[0] if new_models else "default-model")
        
        def refresh_prompt_packs():
            packs = ["None"] + storage.get_prompt_pack_names()
            return gr.Dropdown(choices=packs, value="None")
        
        def update_character_preview(name, age, gender, role, empathy, humor, formality, traits, voice_tone, backstory):
            if not name:
                return ""
            
            try:
                traits_list = [t.strip() for t in traits.split(",") if t.strip()] if traits else []
                character = Character(
                    name=name,
                    age=int(age) if age else 30,
                    gender=gender or "unspecified",
                    role=role,
                    personality={"empathy": empathy, "humor": humor, "formality": formality},
                    traits=traits_list,
                    backstory=backstory or "",
                    voice_tone=voice_tone
                )
                return character.generate_prompt_section()
            except Exception as e:
                return f"Error: {str(e)}"
        
        def update_scenario_preview(setting, time_period, objective, conflict, rules):
            if not setting:
                return ""
            
            try:
                rules_list = [r.strip() for r in rules.split("\n") if r.strip()] if rules else []
                scenario = Scenario(
                    setting=setting,
                    time_period=time_period or "present",
                    objective=objective or "assist the user",
                    conflict=conflict or "",
                    rules=rules_list
                )
                return scenario.generate_prompt_section()
            except Exception as e:
                return f"Error: {str(e)}"
        
        def save_character(name, age, gender, role, empathy, humor, formality, traits, voice_tone, backstory):
            if not name:
                return "Error: Character name is required", gr.Dropdown()
            
            try:
                traits_list = [t.strip() for t in traits.split(",") if t.strip()] if traits else []
                character = Character(
                    name=name,
                    age=int(age) if age else 30,
                    gender=gender or "unspecified",
                    role=role,
                    personality={"empathy": empathy, "humor": humor, "formality": formality},
                    traits=traits_list,
                    backstory=backstory or "",
                    voice_tone=voice_tone
                )
                storage.save_character(character)
                char_names = storage.get_character_names()
                return (
                    f"Character '{name}' saved successfully!",
                    gr.Dropdown(choices=char_names),
                    gr.Dropdown(choices=["None"] + char_names)
                )
            except Exception as e:
                return f"Error saving character: {str(e)}", gr.Dropdown(), gr.Dropdown()
        
        def load_character(char_name):
            if not char_name:
                return [""] * 9
            
            character = storage.load_character(f"{char_name}.json")
            if character:
                traits_str = ", ".join(character.traits)
                return [
                    character.name,
                    character.age,
                    character.gender,
                    character.role,
                    character.personality.get("empathy", 0.5),
                    character.personality.get("humor", 0.5),
                    character.personality.get("formality", 0.5),
                    traits_str,
                    character.voice_tone,
                    character.backstory
                ]
            return [""] * 9
        
        def save_scenario(setting, time_period, objective, conflict, rules):
            if not setting:
                return "Error: Setting is required", gr.Dropdown()
            
            try:
                rules_list = [r.strip() for r in rules.split("\n") if r.strip()] if rules else []
                scenario = Scenario(
                    setting=setting,
                    time_period=time_period or "present",
                    objective=objective or "assist the user",
                    conflict=conflict or "",
                    rules=rules_list
                )
                storage.save_scenario(scenario)
                scenario_names = storage.get_scenario_names()
                return (
                    f"Scenario saved successfully!",
                    gr.Dropdown(choices=scenario_names),
                    gr.Dropdown(choices=["None"] + scenario_names)
                )
            except Exception as e:
                return f"Error saving scenario: {str(e)}", gr.Dropdown(), gr.Dropdown()
        
        def load_scenario(scenario_name):
            if not scenario_name:
                return [""] * 5
            
            scenario = storage.load_scenario(f"{scenario_name}.json")
            if scenario:
                rules_str = "\n".join(scenario.rules)
                return [
                    scenario.setting,
                    scenario.time_period,
                    scenario.objective,
                    scenario.conflict,
                    rules_str
                ]
            return [""] * 5
        
        def update_pack_preview(char_name, scenario_name):
            character = None
            scenario = None
            
            if char_name and char_name != "None":
                character = storage.load_character(f"{char_name}.json")
            
            if scenario_name and scenario_name != "None":
                scenario = storage.load_scenario(f"{scenario_name}.json")
            
            if character or scenario:
                pack = PromptPack(
                    name="Preview",
                    character=character,
                    scenario=scenario
                )
                return pack.generate_system_prompt()
            return ""
        
        def save_prompt_pack(name, description, char_name, scenario_name):
            if not name:
                return "Error: Pack name is required", gr.Dropdown()
            
            try:
                character = None
                scenario = None
                
                if char_name and char_name != "None":
                    character = storage.load_character(f"{char_name}.json")
                
                if scenario_name and scenario_name != "None":
                    scenario = storage.load_scenario(f"{scenario_name}.json")
                
                pack = PromptPack(
                    name=name,
                    description=description or "",
                    character=character,
                    scenario=scenario
                )
                storage.save_prompt_pack(pack)
                return f"Prompt pack '{name}' saved successfully!", gr.Dropdown(choices=storage.get_prompt_pack_names())
            except Exception as e:
                return f"Error saving prompt pack: {str(e)}", gr.Dropdown()
        
        def load_prompt_pack_to_chat(pack_name):
            if not pack_name or pack_name == "None":
                return "You are a helpful, harmless, and honest assistant."
            
            pack = storage.load_prompt_pack(f"{pack_name}.json")
            if pack:
                return pack.generate_system_prompt()
            return "You are a helpful, harmless, and honest assistant."
        
        def optimize_prompt(prompt_text, model):
            """Use LM Studio to optimize the system prompt"""
            if not prompt_text:
                return "", gr.Textbox(visible=False)
            
            try:
                # Create optimization request
                messages = [
                    {"role": "system", "content": "You are an expert at writing and optimizing system prompts for language models. Your task is to enhance the given prompt to be more effective, natural, and engaging while preserving all the key information and personality traits."},
                    {"role": "user", "content": f"Please optimize this system prompt while keeping all the essential information and character traits. Make it more natural and engaging:\n\n{prompt_text}"}
                ]
                
                # Make API call to LM Studio
                completion = chat_bot.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                    top_p=0.9,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stream=False
                )
                
                # Extract optimized prompt
                optimized = completion.choices[0].message.content
                return optimized, gr.Textbox(visible=True)
                
            except Exception as e:
                return f"Error optimizing prompt: {str(e)}", gr.Textbox(visible=False)
        
        # Wire up events
        
        # Chat events
        send_btn.click(
            fn=handle_send,
            inputs=[msg, chatbot, system_msg, model_dropdown, temperature, max_tokens, top_p, freq_penalty, presence_penalty, stream_toggle],
            outputs=[chatbot, msg],
            show_progress=True
        )
        
        msg.submit(
            fn=handle_send,
            inputs=[msg, chatbot, system_msg, model_dropdown, temperature, max_tokens, top_p, freq_penalty, presence_penalty, stream_toggle],
            outputs=[chatbot, msg],
            show_progress=True
        )
        
        clear_btn.click(
            fn=chat_bot.clear_history,
            outputs=[chatbot, msg]
        )
        
        refresh_models_btn.click(
            fn=refresh_models,
            outputs=[model_dropdown]
        )
        
        refresh_packs_btn.click(
            fn=refresh_prompt_packs,
            outputs=[prompt_pack_dropdown]
        )
        
        prompt_pack_dropdown.change(
            fn=load_prompt_pack_to_chat,
            inputs=[prompt_pack_dropdown],
            outputs=[system_msg]
        )
        
        # Character events
        for input_component in [char_name, char_age, char_gender, char_role, char_empathy, char_humor, char_formality, char_traits, char_voice_tone, char_backstory]:
            input_component.change(
                fn=update_character_preview,
                inputs=[char_name, char_age, char_gender, char_role, char_empathy, char_humor, char_formality, char_traits, char_voice_tone, char_backstory],
                outputs=[char_preview]
            )
        
        save_char_btn.click(
            fn=save_character,
            inputs=[char_name, char_age, char_gender, char_role, char_empathy, char_humor, char_formality, char_traits, char_voice_tone, char_backstory],
            outputs=[status, load_char_dropdown, pack_character_dropdown]
        )
        
        load_char_btn.click(
            fn=load_character,
            inputs=[load_char_dropdown],
            outputs=[char_name, char_age, char_gender, char_role, char_empathy, char_humor, char_formality, char_traits, char_voice_tone, char_backstory]
        )
        
        # Scenario events
        for input_component in [scenario_setting, scenario_time, scenario_objective, scenario_conflict, scenario_rules]:
            input_component.change(
                fn=update_scenario_preview,
                inputs=[scenario_setting, scenario_time, scenario_objective, scenario_conflict, scenario_rules],
                outputs=[scenario_preview]
            )
        
        save_scenario_btn.click(
            fn=save_scenario,
            inputs=[scenario_setting, scenario_time, scenario_objective, scenario_conflict, scenario_rules],
            outputs=[status, load_scenario_dropdown, pack_scenario_dropdown]
        )
        
        load_scenario_btn.click(
            fn=load_scenario,
            inputs=[load_scenario_dropdown],
            outputs=[scenario_setting, scenario_time, scenario_objective, scenario_conflict, scenario_rules]
        )
        
        # Prompt pack events
        for input_component in [pack_character_dropdown, pack_scenario_dropdown]:
            input_component.change(
                fn=update_pack_preview,
                inputs=[pack_character_dropdown, pack_scenario_dropdown],
                outputs=[pack_preview]
            )
        
        save_pack_btn.click(
            fn=save_prompt_pack,
            inputs=[pack_name, pack_description, pack_character_dropdown, pack_scenario_dropdown],
            outputs=[status, load_pack_dropdown]
        )
        
        # AI Optimization events
        optimize_prompt_btn.click(
            fn=optimize_prompt,
            inputs=[pack_preview, model_dropdown],
            outputs=[optimized_prompt, optimized_prompt]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_chat_interface()
    
    print("🚀 Starting LM Studio Chat Interface with Personas...")
    print("📋 Make sure LM Studio is running on http://localhost:1234")
    print("🌐 The interface will be available at http://localhost:7861")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True
    )
