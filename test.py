from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """Sentence: {question}. Create a fake story from given context within 30 words.

Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

local_path = (
    "./data/models/ggml-gpt4all-j-v1.3-groovy.bin"  # replace with your desired local file path
)

callbacks = [StreamingStdOutCallbackHandler()]
#llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Three people sitting together"

print(llm_chain.run(question))