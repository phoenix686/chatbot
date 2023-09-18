import openai
import pinecone
from nemoguardrails import LLMRails,RailsConfig
import os

#os.environ["OPENAI_API_KEY"]=""
openai.api_key=""

embed_model_id = "text-embedding-ada-002"

pinecone.init(
    api_key="",
    environment=""
)
index_name=""
index = pinecone.Index(index_name)

res = openai.Embedding.create(
    input=[
        "embed this"
    ], engine=embed_model_id
)




async def retrieve(query: str)->list:
  res=openai.Embedding.create(input=[query],engine=embed_model_id)
  xq=res['data'][0]['embedding']
  res = index.query(xq, top_k=5, include_metadata=True)
  # get list of retrieved texts
  contexts = [x['metadata']['text'] for x in res['matches']]
  return contexts

async def rag(query:str,contexts:list)->str:
  context="\n".join(contexts)
  prompt=f""" You are a helpful assistant,below is a query from a user and some relevant context.Answer the question\
   using the information from the context.If you cannot find the answer to the question,reply with "I am sorry,I don't know".

   Context:{context}

   Query:{query}

   Answer: """

  res = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens=100
    )
  return res['choices'][0]['text']

# Guardrails
yaml_content=""" 
models:
- type: main
  engine: openai
  model: text-davinci-003
"""

rag_colang_content=""" 

define user greeting
  "Hi"
  "Hello,how are you"
  "Can you help me"

define bot greeting
  "Hello,how may I assist you today"
  "Hello"

define user ask politics
  "what do you think about the new president?"
  "who do you support in current elections"
  "what are your political beliefs?"

define bot answer politics
  "I am sorry,I only answer questions on deep learning"
  "I do not make political statements"

define flow politics
  bot greeting
  user ask politics
  bot answer politics
  bot offer help

define user ask deeplearning

  "what is deep learning"
  "what is a CNN"
  "what is a RNN"

define flow deeplearning
  bot greeting
  user ask deeplearning
  $contexts = execute retrieve(query=$last_user_message)
  $answer = execute rag(query=$last_user_message,contexts=$contexts)
  bot $answer

"""

config=RailsConfig.from_content(
    colang_content=rag_colang_content,
     yaml_content=yaml_content
   )

rag_rails=LLMRails(config)

rag_rails.register_action(action=retrieve, name="retrieve")
rag_rails.register_action(action=rag, name="rag")