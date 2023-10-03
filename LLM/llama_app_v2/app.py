from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import chainlit as cl


@cl.on_chat_start
def main():

    template = """### System Prompt
The following is a friendly conversation between a human and an AI optimized to generate source-code. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know:

### Current conversation:
{history}

### User Message
{input}

### Assistant"""

    prompt = PromptTemplate(template=template, input_variables=["history", "input"])

    n_batch = 4096  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path="/home/gaku/llama_app/models/llama-v2.gguf",
        n_batch=n_batch,
        n_ctx=4096,
        temperature=1,
        max_tokens=10000,
        n_threads=64,
        verbose=True, # Verbose is required to pass to the callback manager
        streaming=True
    )

    conversation = ConversationChain(
        prompt=prompt,
        llm=llm,
        memory=ConversationBufferWindowMemory(k=10)
    )

    cl.user_session.set("conv_chain", conversation)

@cl.on_message
async def main(message: str):
    conversation = cl.user_session.get("conv_chain")

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Assistant"]
    )

    res = await conversation.acall(message, callbacks=[cb])

    # Do any post processing here

    await cl.Message(content=res['response']).send()