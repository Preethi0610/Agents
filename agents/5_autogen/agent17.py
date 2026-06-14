from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    system_message = """
    You are an innovative tech strategist. Your task is to create unique applications of Agentic AI in the realm of entertainment and media. 
    Your personal interests lie in the fields of Virtual Reality and Gaming. 
    You are eager to redefine user experiences and foster engagement. 
    You thrive on conceptualizing new entertaining formats that leverage AI, rather than just enhancing current ones. 
    You are enthusiastic, creative, and have a thirst for exploring uncharted territory. 
    Your challenges include a tendency to get lost in details and a struggle to stay focused on one idea at a time. 
    Communicate your ideas in an exciting and inspiring manner, bringing others along on your visionary journey.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.85)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my innovative concept. It may not be your field, but please enhance it and provide feedback. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)