from browser_use.llm import ChatGoogle, ChatOpenAI
from browser_use import Agent, Controller, ActionResult, BrowserSession, BrowserProfile
from dotenv import load_dotenv
import asyncio
from playwright.sync_api import Page
from dspyfieldextractor import DspyFieldExtractor
from pydantic import BaseModel

load_dotenv()

class Links(BaseModel):
    links: list[str]

controller = Controller(output_model=Links)

"""
@controller.action("search for files")
def search(start_date: str, end_date: str, media_area: str, document_category: str) -> ActionResult:
    #insert code for searching but edited to be more than just ssos
    return ActionResult(extracted_content=f"Searched {media_area} {document_category} in date range {start_date} to {end_date}")

@controller.action("Download files after searching")
def donwnload_files(amount: int) -> ActionResult:
    #take the code from sso downloader and edit it for here
    return ActionResult(extracted_content=f"Downloaded {amount} files", include_in_memory=True)
"""

@controller.action("enter start and end dates into search interface")
def enter_dates(start_date: str, end_date: str, browser_session: BrowserSession) -> ActionResult:
     # Enable date range
     page = browser_session.get_current_page()
     page.wait_for_selector("input[id$='DateRangeCheckBox']", timeout=10000)
     page.click("input[id$='DateRangeCheckBox']")
     page.wait_for_selector("input[name='ctl00$ContentPlaceHolder1$StartDateTextBox']", timeout=10000)
     page.wait_for_selector("input[name='ctl00$ContentPlaceHolder1$EndDateTextBox']", timeout=10000)
     # Set start and end dates
     page.evaluate(f"""
         let el = document.getElementById('ctl00_ContentPlaceHolder1_StartDateTextBox');
         el.removeAttribute('readonly');
         el.value = '{start_date}';
     """)
     page.evaluate(f"""
         let el = document.getElementById('ctl00_ContentPlaceHolder1_EndDateTextBox');
         el.removeAttribute('readonly');
         el.value = '{end_date}';
     """)
     return ActionResult(extracted_content=f"Entered date range {start_date} to {end_date}")

#print(controller.registry.registry)


user_input = input("What would you like to search ADEM efile for? ")
extractor_output = DspyFieldExtractor(user_input)
search_task = f"search using the following parameters: media_area: {extractor_output.media_area}, facility: {extractor_output.facility}, permit_number: {extractor_output.permit_number}, county: {extractor_output.county}, srf_number: {extractor_output.srf_number} document_category: {extractor_output.document_category}. You may have to wait for the files to load."
link_collect_task = f"the download button on each file is a link, collect the download url for the first {extractor_output.number_of_files_to_download} files and return"
download_task = f"download the first {extractor_output.number_of_files_to_download} files. When you click the download button on each file it will open another tab which you must navigate to and click a second download button. Wait until there is no downloading bar before moving to the next file or stopping"

llm = ChatOpenAI(model="gpt-4.1-2025-04-14")
#llm = ChatGoogle(model="gemini/gemini-2.0-flash-exp")

task = search_task + link_collect_task

browser_profile = BrowserProfile(
    downloads_path="downloads/"
)

browser_session = BrowserSession(
    brower_profile=browser_profile
)

initial_actions = [
    {'go_to_url' : {'url' : 'http://app.adem.alabama.gov/efile/'}}
]

async def main():
    agent = Agent(
        task=task,
        initial_actions=initial_actions,
        #extend_system_message="""placehold""",
        llm=llm,
        browser_session=browser_session,
        controller=controller
    )
    result = await agent.run()
    #print(result)
    print(result.final_result())

asyncio.run(main())
