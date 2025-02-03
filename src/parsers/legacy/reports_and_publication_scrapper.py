import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
from urllib.parse import urljoin
import requests


class IFSCAScraper:
    def __init__(self):
        self.base_url = "https://ifsca.gov.in"
        self.base_download_dir = "../data"

        # Section URLs and their respective folders
        self.sections = {
            "Committee Report": {
                "url": "/ReportPublication/index/aadg9ruDI%20M=",
                "folder": "committee_reports"
            },
            "Public Consultation": {
                "url": "/ReportPublication/index/sKCVtbX6J9o=",
                "folder": "public_consultations"
            },
            "Annual Reports": {
                "url": "/ReportPublication/index/zcGvy-Iqfcg=",
                "folder": "annual_reports"
            },
            "Bulletin": {
                "url": "/ReportPublication/index/wF6kttc1JR8=",
                "folder": "bulletins"
            },
            "Research": {
                "url": "/ReportPublication/index/mizvnmwVAgs=",
                "folder": "research"
            }
        }

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )

        os.makedirs(self.base_download_dir, exist_ok=True)

    def setup_driver(self):
        """Setup Chrome driver"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        return webdriver.Chrome(options=options)

    def download_file(self, url, folder, filename):
        """Download file using requests"""
        try:
            # Create full URL if it's a relative path
            if url.startswith('../../'):
                url = url.replace('../../', f'{self.base_url}/')

            # Make request to download file
            response = requests.get(url, stream=True, verify=False)
            response.raise_for_status()

            # Create full path for file
            filepath = os.path.join(self.base_download_dir, folder, filename)

            # Save file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logging.info(f"Successfully downloaded: {filename}")
            return True

        except Exception as e:
            logging.error(f"Error downloading {filename}: {str(e)}")
            return False

    def download_documents(self, section_name):
        """Download all documents from a section"""
        try:
            logging.info(f"Starting downloads for section: {section_name}")
            driver = self.setup_driver()
            section_url = urljoin(self.base_url, self.sections[section_name]['url'])
            folder_name = self.sections[section_name]['folder']

            # Create section folder
            os.makedirs(os.path.join(self.base_download_dir, folder_name), exist_ok=True)

            logging.info(f"Navigating to URL: {section_url}")
            driver.get(section_url)
            page_num = 1

            while True:
                logging.info(f"Processing page {page_num}")
                # Wait for table to load
                table = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, 'tblPress'))
                )

                # Process all rows in current page
                rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header

                for row in rows:
                    try:
                        # Extract document info
                        date = row.find_element(By.CSS_SELECTOR, "td.sorting_1").text.strip()
                        title = row.find_element(By.CSS_SELECTOR, "td:nth-child(2)").text.strip()
                        download_link = row.find_element(By.CSS_SELECTOR, "td:nth-child(4) a").get_attribute('href')

                        # Create filename from title and date
                        filename = f"{date.replace('/', '-')}_{title.replace(' ', '_')}.pdf"

                        logging.info(f"Downloading: {filename}")
                        self.download_file(download_link, folder_name, filename)

                    except Exception as e:
                        logging.error(f"Error processing document: {str(e)}")
                        continue

                # Check for next page
                try:
                    next_button = driver.find_element(By.ID, 'tblPress_next')
                    if 'disabled' in next_button.get_attribute('class'):
                        logging.info("Reached last page")
                        break

                    next_button.click()
                    page_num += 1
                    time.sleep(2)
                except Exception as e:
                    logging.info("No more pages found")
                    break

        except Exception as e:
            logging.error(f"Error processing section {section_name}: {str(e)}")
        finally:
            driver.quit()

    def run(self):
        """Run the scraper for all sections"""
        logging.info("Starting IFSCA document scraper")
        for section_name in self.sections:
            logging.info(f"=== Processing section: {section_name} ===")
            self.download_documents(section_name)
            logging.info(f"=== Completed section: {section_name} ===")
            time.sleep(5)  # Wait between sections
        logging.info("Scraping completed")


if __name__ == "__main__":
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    scraper = IFSCAScraper()
    scraper.run()