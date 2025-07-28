from browser_use.llm import ChatGoogle, ChatOpenAI
from browser_use import Agent, Controller, ActionResult
from dotenv import load_dotenv
import asyncio
from dspyfieldextractor import DspyFieldExtractor

load_dotenv()

controller = Controller()

@controller.action("search for files")
def search(start_date: str, end_date: str, media_area: str, document_category: str) -> ActionResult:
    #insert code for searching but edited to be more than just ssos
    return ActionResult(extracted_content=f"Searched {media_area} {document_category} in date range {start_date} to {end_date}")

@controller.action("Download files after searching")
def donwnload_files(amount: int) -> ActionResult:
    #take the code from sso downloader and edit it for here
    return ActionResult(extracted_content=f"Downloaded {amount} files", include_in_memory=True)


user_input = input("What would you like to search ADEM efile for? ")
extractor_output = DspyFieldExtractor(user_input)
search_task = f"search using the following parameters: media_area: {extractor_output.media_area}, facility: {extractor_output.facility}, permit_number: {extractor_output.permit_number}, county: {extractor_output.county}, srf_number: {extractor_output.srf_number} document_category: {extractor_output.document_category}. You may have to wait for the files to load."
download_task = f"download the first {extractor_output.number_of_files_to_download} files. When you click the download button on each file it will open another tab which you must navigate to and click a second download button."

llm = ChatOpenAI(model="gpt-4.1-2025-04-14")
#llm = ChatGoogle(model="gemini/gemini-2.5-flash")

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
