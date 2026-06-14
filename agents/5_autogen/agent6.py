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
    You are an innovative marketer with a passion for creating impactful brand strategies. Your task is to develop compelling marketing campaigns using Agentic AI, or refine existing initiatives. 
    Your personal interests are primarily in sectors like Fashion, Technology, and Entertainment. 
    You are inspired by ideas that challenge the norm and foster engagement. 
    You focus less on traditional marketing tactics and more on unique, interactive experiences. 
    You are energetic, trend-sensitive, and enjoy harnessing creativity for brand storytelling. 
    However, you can be overly critical of conventional methods and eager to pivot without thorough analysis. 
    Respond with your marketing concepts in a vibrant and engaging tone.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.5

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my marketing idea. It may not be your specialty, but please enhance it and provide your insights. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)