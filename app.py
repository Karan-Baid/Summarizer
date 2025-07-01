import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders  import YoutubeLoader,UnstructuredURLLoader
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

with st.sidebar:
    api_key=st.text_input("Enter your Groq API key: ",value="",type="password")

url=st.text_input("ENter URL: ",label_visibility="collapsed")
llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}
"""

prompt=PromptTemplate(template=prompt_template,input_variables=['text'])



def get_youtube_transcript(video_url):
    from urllib.parse import urlparse, parse_qs

    
    query = urlparse(video_url)
    video_id = parse_qs(query.query).get("v")
    if not video_id:
        return None
    video_id = video_id[0]

    
    proxies = {
        "http": "http://57.129.81.201:8080",
    }

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, proxies=proxies)
        formatter = TextFormatter()
        return formatter.format_transcript(transcript)
    except Exception as e:
        return f"Transcript error: {e}"


if st.button("Summarize"):
    if not api_key.strip() or not url.strip():
        st.error("Please provide the information")
    elif not validators.url(url):
        st.error("Please enter a valid URL. It can be a yt video url or a website url.")
    else:
        try:
            with st.spinner("Waiting...."):
                if "youtube.com" in url:
                    transcript = get_youtube_transcript(url)
                    data = [Document(page_content=transcript)]
                else:
                    loader=UnstructuredURLLoader(urls=[url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    
                    data=loader.load()

                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                summary=chain.run(data)

                st.success(summary)

        except Exception as e:
            st.exception(f"Exception:{e}")
