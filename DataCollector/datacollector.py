from browser_use.llm import ChatGoogle, ChatOpenAI
from browser_use import Agent, Controller
from dotenv import load_dotenv
import asyncio
from dspyfieldextractor import DspyFieldExtractor
from pydantic import BaseModel
from downloader import download_pdfs
import os
import shutil

load_dotenv()

#removes files downloaded from previous runs
if os.path.exists("adem_downloads/"):
    shutil.rmtree("adem_downloads/")

class Links(BaseModel):
    links: list[str]

controller = Controller(output_model=Links)

user_input = input("What would you like to search ADEM efile for? ")
extractor_output = DspyFieldExtractor(user_input)
search_task = f"search using the following parameters: media_area: {extractor_output.media_area}, facility: {extractor_output.facility}, permit_number: {extractor_output.permit_number}, county: {extractor_output.county}, srf_number: {extractor_output.srf_number} document_category: {extractor_output.document_category}. You may have to wait for the files to load."
link_collect_task = f"the download button on each file is a link, collect the download url for the first {extractor_output.number_of_files_to_download} files and return"

llm = ChatOpenAI(model="gpt-4.1-2025-04-14")
#llm = ChatGoogle(model="gemini/gemini-2.0-flash-exp")

task = search_task + link_collect_task

initial_actions = [
    {'go_to_url' : {'url' : 'http://app.adem.alabama.gov/efile/'}}
]

async def main():
    agent = Agent(
        task=task,
        initial_actions=initial_actions,
        llm=llm,
        controller=controller
    )
    #result variable is global because asyncio doesn't want playwright running inside the async function and download_pdfs() uses playwright
    global result 
    result = await agent.run()
    result = Links.model_validate_json(result.final_result())
    print(result)
    print()
    print(result.links)


asyncio.run(main())
download_pdfs(result.links)
