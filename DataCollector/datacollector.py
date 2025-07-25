from browser_use.llm import ChatGoogle
from browser_use import Agent 
from dotenv import load_dotenv
import asyncio

load_dotenv()

llm = ChatGoogle(model="gemini-2.5-flash")

async def main():
    agent = Agent(
        task="1.go to http://app.adem.alabama.gov/efile/ 2.search using the following parameters: media area:water, county:mobile, category:inspections. you may have to wait for the files to load 3. download the first file in the list of files. when you click the download button it will open another tab which you must navigate to and press a second download button",
        #extend_system_message="""placehold""",
        llm=llm
    )
    result = await agent.run()
    print(result)

asyncio.run(main())
