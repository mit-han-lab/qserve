"""
This file is a subset of: https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

import base64
import dataclasses
from enum import IntEnum, auto
from io import BytesIO
from typing import Dict, List, Tuple, Union


def get_conv_template_name(model):
    model_path = model.lower()
    if "yi" in model_path:
        return "Yi-34b-chat"
    elif "qwen" in model_path:
        return "qwen-7b-chat"
    elif "vicuna" in model_path:
        return "vicuna_v1.1"
    elif "llama-3" in model_path or "llama3" in model_path:
        return "llama-3"
    elif "llama" in model_path:
        return "llama-2"
    else:
        return "zero_shot"


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    LLAMA3 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    DEEPSEEK_CHAT = auto()
    METAMATH = auto()
    YUAN2 = auto()
    GEMMA = auto()
    CLLM = auto()
    DEFAULT = auto()


IMAGE_PLACEHOLDER_STR = "$$<image>$$"


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    # Each message is either a string or a tuple of (string, List[image_url]).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA3:
            ret = "<|begin_of_text|>"
            if self.system_message:
                ret += system_prompt
            else:
                ret += ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    ret += f"{message.strip()}<|eot_id|>"
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ""
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += "<s>"
                if message:
                    ret += role + ":" + message + seps[i % 2] + "\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ""
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.METAMATH:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for i, (role, message) in enumerate(self.messages):
                # For MetaMath, sep2 is used to prefix the message.
                starting_sep = ":\n" if i % 2 == 0 else ": " + self.sep2
                ending_sep = self.sep if i % 2 == 0 else ""
                if message:
                    ret += role + starting_sep + message + ending_sep
                else:
                    ret += role + starting_sep
            return ret
        elif self.sep_style == SeparatorStyle.DEEPSEEK_CHAT:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.YUAN2:
            seps = [self.sep, self.sep2]
            ret = ""
            if self.system_message:
                ret += system_prompt + seps[1]
            for _, message in self.messages:
                if message:
                    ret += message + "<n>"
                else:
                    ret += ""
            ret = ret.rstrip("<n>") + seps[0]
            return ret
        elif self.sep_style == SeparatorStyle.GEMMA:
            ret = "<bos>"
            for role, message in self.messages:
                if message:
                    ret += "<start_of_turn>" + role + "\n" + message + self.sep
                else:
                    ret += "<start_of_turn>" + role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.CLLM:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages[-2:]):
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DEFAULT:
            ret = system_prompt + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + "\n"
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def get_images(self):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    for image in msg[1]:
                        images.append(image)

        return images

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def get_system_message(self):
        """return the system message."""
        return self.system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def convert_image_to_base64(self, image):
        """Given an image, return the base64 encoded image string."""
        import requests
        from PIL import Image

        # Load image if it has not been loaded in yet
        if type(image) == str:
            if image.startswith("http://") or image.startswith("https://"):
                response = requests.get(image)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            elif "base64" in image:
                # OpenAI format is: data:image/jpeg;base64,{base64_encoded_image_str}
                return image.split(",")[1]
            else:
                image = Image.open(image).convert("RGB")

        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 2048, 2048
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if longest_edge != max(image.size):
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()

        return img_b64_str

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image = msg
                    img_b64_str = image[0]  # Only one image on gradio at one time
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace("<image>\n", "").strip()

                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def extract_text_from_messages(self):
        return [
            (role, message[0]) if type(message) is tuple else (role, message)
            for role, message in self.messages
        ]

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.extract_text_from_messages(),
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# An empty template for raw conversation.
register_conv_template(
    Conversation(
        name="raw",
        system_message="",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
    )
)

# A template with a one-shot conversation example

# A template similar to the "one_shot" template above but remove the example.
register_conv_template(
    Conversation(
        name="zero_shot",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# llama2 template
# reference: https://huggingface.co/blog/codellama#conversational-instructions
# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)

# llama3 template
# reference: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
# reference: https://github.com/meta-llama/llama3/blob/0cee08ec68f4cfc0c89fe4a9366d82679aaa2a66/llama/tokenizer.py#L222
register_conv_template(
    Conversation(
        name="llama-3",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.LLAMA3,
        sep="",
        stop_str="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    )
)

# NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO template
# source: https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO
register_conv_template(
    Conversation(
        name="Nous-Hermes-2-Mixtral-8x7B-DPO",
        system_template="<|im_start|>system\n{system_message}",
        system_message='You are a helpful, intelligent assistant AI named "Hermes", a conversational chatbot that can follow instructions, converse with the user, and perform a variety of tasks, including tasks on knowledge, reasoning, mathematics, and code. Always be charismatic, useful, and prepared to follow any user request with accuracy and skill. You should respond with high quality, fluent, and detailed responses. Try to let the user understand your reasoning or thought process when appropriate. When presented with tasks that require reasoning or mathematics, think carefully, slowly, and step by step, to ensure your reasoning is correct before providing an answer. Utilize the "Examples" section to assist you in performing the task. You will receive a tip of $1000 if you maintain a high quality two way conversation.',
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32000, 32001],
    )
)

# Qwen-chat default template
# source: https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py#L130
register_conv_template(
    Conversation(
        name="qwen-7b-chat",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    )
)

# source: https://huggingface.co/01-ai/Yi-34B-Chat/blob/main/tokenizer_config.json#L60
register_conv_template(
    Conversation(
        name="Yi-34b-chat",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            2,
            6,
            7,
            8,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>"
        stop_str="<|endoftext|>",
    )
)
