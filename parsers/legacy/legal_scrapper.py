import os
import time
import logging
import requests
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import urljoin


class SimplifiedIFSCAScraper:
    def __init__(self):
        self.base_url = "https://ifsca.gov.in"
        self.base_download_dir = "data/legal"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('legal_scraper.log'),
                logging.StreamHandler()
            ]
        )

        # Setup Chrome options
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')

        # Create base download directory
        os.makedirs(self.base_download_dir, exist_ok=True)

    def normalize_folder_name(self, folder_name):
        """Convert folder name to lowercase and replace spaces with underscores"""
        # Replace spaces and special characters with underscores
        normalized = folder_name.strip().lower()
        normalized = ''.join(c if c.isalnum() or c == ' ' else '_' for c in normalized)
        normalized = normalized.replace(' ', '_')
        # Remove multiple consecutive underscores
        while '__' in normalized:
            normalized = normalized.replace('__', '_')
        return normalized.strip('_')

    def start_driver(self):
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.driver.implicitly_wait(10)

    def close_driver(self):
        if hasattr(self, 'driver'):
            self.driver.quit()

    def download_file(self, url, folder, filename):
        """Download file using requests"""
        try:
            # Create full URL if it's a relative path
            if url.startswith('/'):
                url = urljoin(self.base_url, url)

            # Normalize folder name
            normalized_folder = self.normalize_folder_name(folder)

            # Make request to download file
            response = requests.get(url, stream=True, verify=False)
            response.raise_for_status()

            # Create folder if it doesn't exist
            folder_path = os.path.join(self.base_download_dir, normalized_folder)
            os.makedirs(folder_path, exist_ok=True)

            # Create full path for file using os.path.join with normalized path
            filepath = os.path.normpath(os.path.join(folder_path, filename))

            # Save file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logging.info(f"Successfully downloaded: {filename} to {normalized_folder}")
            return True

        except Exception as e:
            logging.error(f"Error downloading {filename} to {normalized_folder}: {str(e)}")
            return False

    def get_sections(self):
        """Get all legal sections from the left panel"""
        self.driver.get(f"{self.base_url}/Legal/Index/sKCVtbX6J9o=")
        sections = []

        try:
            left_panel = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'left-panel'))
            )

            nav_links = left_panel.find_elements(By.CLASS_NAME, 'nav-link')

            for link in nav_links:
                section_name = link.text.strip()
                section_url = link.get_attribute('href')
                if section_url:
                    sections.append({'name': section_name, 'url': section_url})
                    logging.info(f"Found section: {section_name}")

        except Exception as e:
            logging.error(f"Error getting sections: {str(e)}")

        return sections

    def process_page(self, section_name):
        """Process a single page of documents"""
        try:
            # Wait for table to load
            table = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, 'GridList'))
            )

            rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header row

            for row in rows:
                try:
                    # Extract document info
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 4:
                        date = cells[0].text.strip()
                        title = cells[1].text.strip()
                        download_link = cells[3].find_element(By.TAG_NAME, "a").get_attribute('href')

                        if download_link and download_link.endswith('.pdf'):
                            # Create filename from date and title
                            safe_title = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in title)
                            filename = f"{date.replace('/', '-')}_{safe_title}.pdf"

                            logging.info(f"Downloading: {filename}")
                            self.download_file(download_link, section_name, filename)
                            time.sleep(1)  # Small delay between downloads

                except Exception as e:
                    logging.error(f"Error processing document: {str(e)}")
                    continue

        except Exception as e:
            logging.error(f"Error processing page: {str(e)}")

    def has_next_page(self):
        """Check if there is a next page available"""
        try:
            next_button = self.driver.find_element(By.ID, 'GridList_next')
            return 'disabled' not in next_button.get_attribute('class')
        except NoSuchElementException:
            return False

    def click_next_page(self):
        """Click the next page button"""
        try:
            next_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'GridList_next'))
            )
            next_button.click()
            time.sleep(2)  # Wait for page to load
            return True
        except Exception as e:
            logging.error(f"Error clicking next page: {str(e)}")
            return False

    def download_documents(self):
        """Main method to download all legal documents"""
        try:
            self.start_driver()
            sections = self.get_sections()
            logging.info(f"Found {len(sections)} sections")

            for section in sections:
                logging.info(f"=== Processing section: {section['name']} ===")

                # Navigate to section
                self.driver.get(section['url'])

                page_number = 1
                while True:
                    logging.info(f"Processing page {page_number} of section {section['name']}")
                    self.process_page(section['name'])

                    if self.has_next_page():
                        if not self.click_next_page():
                            break
                        page_number += 1
                    else:
                        break

                time.sleep(2)  # Wait between sections
                logging.info(f"=== Completed section: {section['name']} ===")

        except Exception as e:
            logging.error(f"Error: {str(e)}")
        finally:
            self.close_driver()


if __name__ == "__main__":
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    scraper = SimplifiedIFSCAScraper()
    scraper.download_documents()