from typing import List
from openai import OpenAI


class LMStudioChat:
    """Wrapper around OpenAI client for LM Studio."""

    def __init__(self) -> None:
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.conversation_history: List[dict] = []

    def get_available_models(self) -> List[str]:
        """Return available models from LM Studio."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:  # pragma: no cover - network dependent
            print(f"Error fetching models: {e}")
            return ["default-model"]

    def clear_history(self) -> tuple[list, str]:
        """Clear the conversation history."""
        self.conversation_history = []
        return [], ""

    def chat_with_lm_studio(
        self,
        message: str,
        history: List[dict],
        system_message: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
    ) -> tuple[list, str]:
        """Send a message and return updated history."""
        try:
            messages = []
            if system_message.strip():
                messages.append({"role": "system", "content": system_message})
            for msg in history:
                messages.append(msg)
            messages.append({"role": "user", "content": message})
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=False,
            )
            response = completion.choices[0].message.content
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history, ""
        except Exception as e:  # pragma: no cover - network dependent
            error_msg = f"Error: {e}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""

    def stream_chat_with_lm_studio(
        self,
        message: str,
        history: List[dict],
        system_message: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
    ):
        """Yield streaming responses from LM Studio."""
        try:
            messages = []
            if system_message.strip():
                messages.append({"role": "system", "content": system_message})
            for msg in history:
                messages.append(msg)
            messages.append({"role": "user", "content": message})
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": ""})
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=True,
            )
            partial_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    partial_response += chunk.choices[0].delta.content
                    history[-1]["content"] = partial_response
                    yield history, ""
        except Exception as e:  # pragma: no cover - network dependent
            error_msg = f"Error: {e}"
            if history and history[-1]["role"] == "assistant":
                history[-1]["content"] = error_msg
            else:
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": error_msg})
            yield history, ""
