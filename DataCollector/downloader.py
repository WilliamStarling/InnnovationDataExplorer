from playwright.sync_api import sync_playwright
import os 
import logging
import time
import sys

def download_pdfs(links: list[str], limit: int = None) -> None:
    """Download each PDF to ``DOWNLOAD_DIR`` using Playwright."""
    DOWNLOAD_DIR = "downloads/"
    with sync_playwright() as pw:
        DEV_MODE = "--show" in sys.argv
        browser = pw.chromium.launch(headless=not DEV_MODE, slow_mo=250 if DEV_MODE else 0)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        count = 0
        for link in links:
            print(link)
            if limit is not None and count >= limit:
                break
            dest = DOWNLOAD_DIR + str(count) + ".pdf"
            if os.path.exists(dest):
                continue
            logging.info("Downloading %s", dest)
            try:
                page.goto(link, timeout=60000)
                # Wait for the download button to be visible
                page.wait_for_selector("#STR_DOWNLOAD", timeout=30000, state="visible")
                with page.expect_download() as download_info:
                    page.click("#STR_DOWNLOAD")
                download = download_info.value
                download.save_as(dest)
                count += 1
                time.sleep(0.5)
            except Exception as e:
                logging.warning("Failed to download %s: %s", link, e)
        context.close()
        browser.close()
        
#{"links": ["http://lf.adem.alabama.gov/weblink/DocView.aspx?id=105713248&dbid=0", "http://lf.adem.alabama.gov/weblink/DocView.aspx?id=105713249&dbid=0"]}