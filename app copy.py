import gradio as gr
import openai
from openai import OpenAI
import json
import asyncio
from typing import List, Tuple, Optional

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
    """Create the Gradio chat interface"""
    chat_bot = LMStudioChat()
    
    # Get available models
    available_models = chat_bot.get_available_models()
    
    with gr.Blocks(title="LM Studio Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 LM Studio Chat Interface")
        gr.Markdown("Advanced chat interface with parameter controls for LM Studio")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    container=True,
                    bubble_full_width=False
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
                
                # System message
                system_msg = gr.Textbox(
                    label="System Message",
                    placeholder="You are a helpful assistant...",
                    lines=3,
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
                    info="Controls randomness (0=deterministic, 2=very random)"
                )
                
                # Max tokens
                max_tokens = gr.Slider(
                    minimum=1,
                    maximum=4096,
                    value=1024,
                    step=1,
                    label="Max Tokens",
                    info="Maximum number of tokens to generate"
                )
                
                # Top-p
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p",
                    info="Nucleus sampling parameter"
                )
                
                # Frequency penalty
                freq_penalty = gr.Slider(
                    minimum=-2.0,
                    maximum=2.0,
                    value=0.0,
                    step=0.1,
                    label="Frequency Penalty",
                    info="Penalize repeated tokens"
                )
                
                # Presence penalty
                presence_penalty = gr.Slider(
                    minimum=-2.0,
                    maximum=2.0,
                    value=0.0,
                    step=0.1,
                    label="Presence Penalty",
                    info="Penalize new topics"
                )
                
                # Refresh models button
                refresh_btn = gr.Button("🔄 Refresh Models", variant="secondary")
        
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
        
        # Wire up events
        send_event = send_btn.click(
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
        
        refresh_btn.click(
            fn=refresh_models,
            outputs=[model_dropdown]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_chat_interface()
    
    print("🚀 Starting LM Studio Chat Interface...")
    print("📋 Make sure LM Studio is running on http://localhost:1234")
    print("🌐 The interface will be available at http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
