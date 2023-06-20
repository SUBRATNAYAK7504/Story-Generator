from options import *

#this function is for OPENAI story generation
def create_story_using_chatgpt(scenario):
    if VERBOSE_MODE: print("Story is being generated using chatgpt")

    template = """
    You are a story teller.
    Generate a simple intresting story based on the scenario not more than 20 words.

    Scenario: {scenario}
    """
    prompt = PromptTemplate(template = template, input_variables = ["scenario"])
    story_llm = LLMChain(prompt = prompt, verbose = True, llm = OpenAI(model_name = "gpt-3.5-turbo", temperature = 1))
    
    story = story_llm.predict(scenario = scenario)

    if VERBOSE_MODE: print("Story generated successfully")
    if VERBOSE_MODE: print(f"story is: {story}")

    return story

#this function is for falcon story generation
def create_story_using_falcon(scenario):
    if VERBOSE_MODE: print("Story is being generated using falcon")
    model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

    tokenizer = AutoTokenizer.from_pretrained(model)

    huggingface_pipeline = pipeline(
        "text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline = huggingface_pipeline, model_kwargs = {'temperature':0})
    
    template = """
    You are a story teller.
    Generate a simple intresting story based on the scenario not more than 20 words.

    Scenario: {scenario}
    """
    prompt = PromptTemplate(template = template, input_variables = ["scenario"])
    story_llm = LLMChain(prompt = prompt, verbose = True, llm = llm)
    story = story_llm.run(scenario)

    if VERBOSE_MODE: print("Story generated successfully")
    if VERBOSE_MODE: print(f"story is: {story}")

    return story

#this function is for gpt4-all story generation
def create_story_using_gpt4_all(scenario):
    if VERBOSE_MODE: print("Story is being generated using GPT4ALL")
    model = "nomic-ai/gpt4all-j" #tiiuae/falcon-40b-instruct

    #tokenizer = AutoTokenizer.from_pretrained(model)

    huggingface_pipeline = pipeline(
        "text-generation", #task
        model=model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        
    )
    llm = HuggingFacePipeline(pipeline = huggingface_pipeline, model_kwargs = {'temperature':0})
    
    template = """
    You are a story teller.
    Generate a simple intresting story based on the scenario not more than 20 words.

    Scenario: {scenario}
    """
    prompt = PromptTemplate(template = template, input_variables = ["scenario"])
    story_llm = LLMChain(prompt = prompt, verbose = True, llm = llm)
    story = story_llm.run(scenario)

    if VERBOSE_MODE: print("Story generated successfully")
    if VERBOSE_MODE: print(f"story is: {story}")

    return story

#this function uses downloaded model
def create_story_using_gpt4_all_local(scenario):
    if VERBOSE_MODE: print("Story is being generated using GPT4ALL Local")

    template = """
        {question}. 
        Create a story from above context.
        
        Answer: """

    prompt = PromptTemplate(template=template, input_variables=["question"])

    local_path = (
        "./data/models/ggml-gpt4all-j-v1.3-groovy.bin"  # replace with your desired local file path
    )

    callbacks = [StreamingStdOutCallbackHandler()]
    #llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
    llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True)

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    story = llm_chain.run(scenario)
    if VERBOSE_MODE: print("Story generated successfully")
    if VERBOSE_MODE: print(f"story is: {story}")

    return story
    

