from browser_use.llm import ChatGoogle
from browser_use import Agent 
from dotenv import load_dotenv
import asyncio
from dspyfieldextractor import DspyFieldExtractor

load_dotenv()

llm = ChatGoogle(model="gemini-2.5-flash")

user_input = input("What would you like to search ADEM efile for")
extractor_output = DspyFieldExtractor(user_input)
search_task = f"search using the following parameters: media_area: {extractor_output.media_area}, facility: {extractor_output.facility}, permit_number: {extractor_output.permit_number}, county: {extractor_output.county}, srf_number: {extractor_output.srf_number} document_category: {extractor_output.document_category}. You may have to wait for the files to load."
download_task = f"download the first {extractor_output.number_of_files_to_download} files. When you click the download button on each file it will open another tab which you must navigate to and click a second download button."

task = search_task + download_task

initial_actions = [
    {'go_to_url' : {'url' : 'http://app.adem.alabama.gov/efile/'}}
]

async def main():
    agent = Agent(
        task=task,
        initial_actions=initial_actions,
        #extend_system_message="""placehold""",
        llm=llm
    )
    result = await agent.run()
    print(result)

asyncio.run(main())
