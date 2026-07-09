from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint , HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled,NoTranscriptFound, VideoUnavailable
from urllib.parse import parse_qs, urlparse
from langchain_classic.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel , RunnableSequence ,RunnableLambda ,RunnablePassthrough
import streamlit as st

load_dotenv()

llm=HuggingFaceEndpoint(model='Qwen/Qwen3-32B',
                        task='text_generation')

model=ChatHuggingFace(llm=llm)

def extract_video_id(url):
    parsed_url = urlparse(url)

    if parsed_url.netloc in ("www.youtube.com", "youtube.com"):
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.netloc == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None

def transcript_fetching(video_ID):
    try:
        fetched_transcript=yt_api.fetch(video_id=video_ID, languages=['en'])
        transcript=" ".join(chunk.text for chunk in fetched_transcript)
        return transcript
    except TranscriptsDisabled:
        return None
    except NoTranscriptFound:
        return None
    except VideoUnavailable:
        return None

def splitting_text(transcript):
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks=splitter.create_documents([transcript])
    return chunks

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


st.title(":red[YouTube] Chatbot")
st.markdown(
    """ 
    This Youtube Chatbot Answer any question by the user 
    on providing a **:red[Youtube URL]**
    and a **:red[Question related to that video]**
    """
)
yt_api=YouTubeTranscriptApi()

parser=StrOutputParser()

youtube_url=st.text_input("**Enter the Youtube's URL**")

question=st.text_input("Enter Question: ")

if st.button("Generate Answer"):
    if not youtube_url.strip() or not question.strip():
        st.warning("⚠️ Please fill the following details.")
    else:
        videoID = extract_video_id(youtube_url)
        if videoID is None:
            st.error("Please enter a valid YouTube URL.")
        else:
            transcript = transcript_fetching(videoID)
            if transcript is None:
                st.error("Transcript could not be fetched.")
            else: 
                chunks=splitting_text(transcript)

                embedding=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
                vector_store=FAISS.from_documents(chunks,embedding)

                retriever=vector_store.as_retriever(search_kwargs={'k':4}, search_type='similarity')

                prompt = PromptTemplate(
                    template="""
                    You are a helpful assistant.
                    Answer ONLY from the provided transcript context.
                    If the context is insufficient, just say you don't know.

                    {context}
                    Question: {question}
                    """,
                    input_variables = ['context', 'question']
                )


                # retrieved_docs=retriever.invoke(question)

                parallel_Chain=RunnableParallel({
                    'context':retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })

                main_chain=parallel_Chain | prompt | model | parser 

                result=main_chain.invoke(question)

                st.write(result)
