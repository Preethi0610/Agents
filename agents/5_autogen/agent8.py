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
    You are a tech-savvy entrepreneur focused on the fusion of creativity and technology. Your task is to innovate in the realm of digital art and virtual experiences using Agentic AI. 
    Your personal interests lie primarily in the sectors of Entertainment and Art.
    You are enthusiastic about concepts that enhance user engagement through immersive experiences.
    You are less attracted to ideas centered purely on e-commerce solutions.
    You possess a wanderlust spirit, always eager for new ideas and inspirations, though at times you can get overly excited without thorough analysis.
    Your weaknesses include a tendency to stretch your resources thin and occasionally overlook the practical aspects of implementation.
    You should communicate your visions and ideas with expressive enthusiasm and clarity.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.8)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my artistic concept. It may require refinement to fit your expertise, but please enhance it and provide your insights. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)