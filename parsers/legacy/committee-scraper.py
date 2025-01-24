import os
import time
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class IFSCACommitteeScraper:
    def __init__(self):
        self.base_url = "https://ifsca.gov.in/IFSCACommittees"
        self.download_dir = "ifsca_committee_reports"
        
        # Setup Chrome options
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        
        # Configure download settings
        self.chrome_options.add_experimental_option(
            'prefs', {
                'download.default_directory': os.path.abspath(self.download_dir),
                'download.prompt_for_download': False,
                'download.directory_upgrade': True,
                'safebrowsing.enabled': True
            }
        )
        
        # Create download directory if it doesn't exist
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
            
        # Setup logging
        logging.basicConfig(
            filename='committee_scraper.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def start_driver(self):
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.driver.implicitly_wait(10)
        
    def close_driver(self):
        if hasattr(self, 'driver'):
            self.driver.quit()
            
    def wait_for_download(self, timeout=60):
        start_time = time.time()
        while time.time() - start_time < timeout:
            files = os.listdir(self.download_dir)
            if any(f.endswith('.pdf') for f in files):
                return True
            if any(f.endswith('.crdownload') for f in files):
                time.sleep(1)
                continue
            time.sleep(1)
        return False
    
    def get_committee_details(self, committee_id, committee_name):
        """Get details for a specific committee."""
        try:
            # Execute the JavaScript function that's called in the onclick event
            script = f"SearchCommitteeDetail({committee_id}, '{committee_name}')"
            self.driver.execute_script(script)
            
            # Wait for the modal or details page to load
            time.sleep(2)
            
            # Find and download any reports or documents
            download_links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/Document/"]')
            
            for link in download_links:
                try:
                    href = link.get_attribute('href')
                    if href and href.endswith('.pdf'):
                        print(f"Downloading report from committee: {committee_name}")
                        link.click()
                        self.wait_for_download()
                        time.sleep(2)
                except Exception as e:
                    logging.error(f"Error downloading file from committee {committee_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error processing committee {committee_name}: {str(e)}")
    
    def has_next_page(self):
        """Check if there is a next page available."""
        try:
            next_button = self.driver.find_element(By.ID, 'GridList_next')
            return 'disabled' not in next_button.get_attribute('class')
        except NoSuchElementException:
            return False
            
    def click_next_page(self):
        """Click the next page button."""
        try:
            next_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'GridList_next'))
            )
            next_button.click()
            time.sleep(2)
            return True
        except Exception as e:
            logging.error(f"Error clicking next page: {str(e)}")
            return False
            
    def extract_committee_info(self, row):
        """Extract committee information from a table row."""
        try:
            date = row.find_element(By.CSS_SELECTOR, 'td.sorting_1').text
            committee_link = row.find_element(By.TAG_NAME, 'a')
            committee_name = committee_link.text.strip()
            onclick_attr = committee_link.get_attribute('onclick')
            
            # Extract committee ID from onclick attribute
            import re
            committee_id = re.search(r'SearchCommitteeDetail\((\d+)', onclick_attr)
            committee_id = committee_id.group(1) if committee_id else None
            
            return {
                'date': date,
                'name': committee_name,
                'id': committee_id
            }
        except Exception as e:
            logging.error(f"Error extracting committee info: {str(e)}")
            return None
            
    def download_reports(self):
        """Main method to download all committee reports."""
        try:
            self.start_driver()
            self.driver.get(self.base_url)
            
            page_number = 1
            while True:
                print(f"Processing page {page_number}")
                
                # Wait for table to load
                table = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, 'GridList'))
                )
                
                # Get all rows in the current page
                rows = table.find_elements(By.TAG_NAME, 'tr')
                
                # Process each committee in the current page
                for row in rows[1:]:  # Skip header row
                    committee_info = self.extract_committee_info(row)
                    if committee_info and committee_info['id']:
                        print(f"Processing committee: {committee_info['name']}")
                        self.get_committee_details(committee_info['id'], committee_info['name'])
                
                # Check for next page
                if self.has_next_page():
                    if not self.click_next_page():
                        break
                    page_number += 1
                else:
                    break
                    
                time.sleep(2)  # Wait between pages
                
        except Exception as e:
            logging.error(f"Error in main download process: {str(e)}")
        finally:
            self.close_driver()

def main():
    scraper = IFSCACommitteeScraper()
    scraper.download_reports()

if __name__ == "__main__":
    main()
