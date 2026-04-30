from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


def get_message_content(msg) -> str:
    if isinstance(msg, BaseMessage):
        return msg.content
    elif isinstance(msg, dict):
        return msg.get("content", "")
    return str(msg)


def get_message_role(msg) -> str:
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, AIMessage):
        return "assistant"
    elif isinstance(msg, dict):
        return msg.get("role", "user")
    return "user"
