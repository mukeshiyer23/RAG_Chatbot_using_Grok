import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, unquote
import logging
from datetime import datetime


class IFSCADocumentDownloader:
    def __init__(self):
        self.base_url = "https://ifsca.gov.in"
        self.legal_base_url = "https://ifsca.gov.in/Legal/Index"
        self.download_dir = "ifsca_documents"
        self.links_log_file = "downloaded_links.txt"
        self.setup_logging()
        self.session = requests.Session()

    def setup_logging(self):
        """Set up logging configuration."""
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

        log_file = os.path.join(self.download_dir, f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def get_sections(self):
        """Get all available sections from the left panel."""
        try:
            response = self.session.get(f"{self.base_url}/Legal/Index/sKCVtbX6J9o=")  # Using Act as starting point
            if response.status_code != 200:
                logging.error("Failed to access the legal page")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            left_panel = soup.find('div', {'class': 'left-panel'})
            sections = []

            if left_panel:
                for link in left_panel.find_all('a', {'class': 'nav-link'}):
                    section_name = link.text.strip()
                    section_url = link.get('href')
                    if section_url and not section_url.startswith(('http://', 'https://')):
                        section_url = urljoin(self.base_url, section_url)
                    sections.append({
                        'name': section_name,
                        'url': section_url
                    })

            return sections
        except Exception as e:
            logging.error(f"Error getting sections: {str(e)}")
            return []

    def clean_filename(self, filename):
        """Clean filename to be valid and within Windows path limits."""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        filename = filename.replace('(', '_').replace(')', '_')
        filename = ' '.join(filename.split())  # normalize spaces
        filename = '_'.join(filter(None, filename.split('_')))  # normalize underscores

        max_length = 100
        if len(filename) > max_length:
            name_parts = filename.rsplit('.', 1) if '.' in filename else [filename, '']
            base_name = name_parts[0]
            extension = f".{name_parts[1]}" if len(name_parts) > 1 else ""
            available_space = max_length - len(extension) - 3
            half_length = available_space // 2
            filename = f"{base_name[:half_length]}...{base_name[-half_length:]}{extension}"

        return filename

    def log_link(self, url, title, date):
        """Log the downloaded link for tracking."""
        with open(self.links_log_file, 'a') as log_file:
            log_file.write(f"{date} | {title} | {url}\n")

    def download_file(self, url, title, date, section):
        """Enhanced download function with better path handling."""
        try:
            section_dir = os.path.join(self.download_dir, self.clean_filename(section))
            if not os.path.exists(section_dir):
                os.makedirs(section_dir)

            try:
                date_str = datetime.strptime(date, "%d/%m/%Y").strftime("%Y%m%d")
            except ValueError:
                date_str = datetime.now().strftime("%Y%m%d")

            title_words = title.split()[:5]
            short_title = '_'.join(title_words)
            if len(title_words) < len(title.split()):
                short_title += '_etc'

            filename = f"{date_str}_{self.clean_filename(short_title)}.pdf"
            file_path = os.path.join(section_dir, filename)

            if len(file_path) >= 260:
                short_title = '_'.join(title_words[:3])
                filename = f"{date_str}_{self.clean_filename(short_title)}_doc.pdf"
                file_path = os.path.join(section_dir, filename)

            if os.path.exists(file_path):
                logging.info(f"File already exists: {filename} in section {section}")
                return True

            if not url.startswith(('http://', 'https://')):
                download_url = urljoin(self.base_url, url)
            else:
                download_url = url

            download_url = unquote(download_url)

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.session.get(download_url, stream=True, headers=headers, timeout=30)
                    if response.status_code == 200:
                        with open(file_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        self.log_link(download_url, title, date)
                        logging.info(f"Successfully downloaded: {filename} to {section}")
                        return True
                    else:
                        logging.error(f"Failed to download {filename}: Status code {response.status_code}")
                        time.sleep(2 * (attempt + 1))
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Failed to download {filename} after {max_retries} attempts: {str(e)}")
                        return False
                    time.sleep(2 * (attempt + 1))

            return False

        except Exception as e:
            logging.error(f"Error downloading {title}: {str(e)}")
            return False

    def get_total_pages(self, page_html):
        """Extract total number of pages from pagination."""
        soup = BeautifulSoup(page_html, 'html.parser')
        pagination = soup.find('ul', {'class': 'pagination'})
        if pagination:
            pages = pagination.find_all('li', {'class': 'paginate_button'})
            # Filter out 'Previous' and 'Next' buttons
            page_numbers = [int(page.text) for page in pages if page.text.isdigit()]
            return max(page_numbers) if page_numbers else 1
        return 1

    def download_all_sections(self):
        """Download documents from all available sections."""
        sections = self.get_sections()
        logging.info(f"Found {len(sections)} sections to process")

        for section in sections:
            self.download_section_documents(section)
            # Add delay between sections
            time.sleep(3)

    def get_document_links(self, page_html):
        """Extract document information from the page with enhanced link detection."""
        soup = BeautifulSoup(page_html, 'html.parser')
        documents = []

        # Try different table structures
        table = soup.find('table', {'id': 'GridList'}) or soup.find('table', {'class': 'table'})

        if table:
            # Handle standard table layout
            for row in table.find_all('tr')[1:]:  # Skip header row
                cols = row.find_all('td')
                if len(cols) >= 4:
                    date = cols[0].text.strip().replace('\n', '').split()[-1]
                    title = cols[1].text.strip()

                    # Try multiple possible locations for download link
                    download_link = None
                    for col in cols:
                        link = col.find('a')
                        if link and 'href' in link.attrs:
                            href = link['href']
                            if 'Document' in href or href.endswith('.pdf'):
                                download_link = link
                                break

                    if download_link:
                        documents.append({
                            'date': date,
                            'title': title,
                            'url': download_link['href']
                        })
        else:
            # Handle alternative layouts (like lists or divs)
            content_div = soup.find('div', {'class': ['content-area', 'main-content']})
            if content_div:
                # Look for links within content area
                for item in content_div.find_all(['div', 'li']):
                    # Try to find date and title
                    date_elem = item.find('span', {'class': 'date'}) or item.find(
                        text=lambda t: t and any(m in t for m in ['/20', '.20']))
                    date = date_elem.text.strip() if date_elem else datetime.now().strftime("%d/%m/%Y")

                    # Find title and link
                    link = item.find('a')
                    if link and 'href' in link.attrs:
                        title = link.text.strip()
                        if not title:
                            # Try to get title from parent element
                            title = item.get_text().strip()

                        if title:
                            documents.append({
                                'date': date,
                                'title': title,
                                'url': link['href']
                            })

        return documents

    def download_section_documents(self, section):
        """Enhanced section document download with better error handling."""
        try:
            logging.info(f"Processing section: {section['name']}")

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/91.0.4472.124 Safari/537.36',
            }

            response = self.session.get(section['url'], headers=headers, timeout=30)
            if response.status_code != 200:
                logging.error(f"Failed to access section: {section['name']}")
                return

            total_pages = self.get_total_pages(response.text)
            logging.info(f"Found {total_pages} pages in section {section['name']}")

            for page in range(1, total_pages + 1):
                logging.info(f"Processing page {page} of {total_pages} in section {section['name']}")

                if page == 1:
                    page_html = response.text
                else:
                    page_url = f"{section['url']}?page={page}"
                    response = self.session.get(page_url, headers=headers, timeout=30)
                    if response.status_code != 200:
                        logging.error(f"Failed to access page {page} in section {section['name']}")
                        continue
                    page_html = response.text

                documents = self.get_document_links(page_html)
                logging.info(f"Found {len(documents)} documents on page {page} in section {section['name']}")

                for doc in documents:
                    self.download_file(doc['url'], doc['title'], doc['date'], section['name'])
                    time.sleep(1)

                time.sleep(2)

        except Exception as e:
            logging.error(f"Error processing section {section['name']}: {str(e)}")


def main():
    downloader = IFSCADocumentDownloader()
    downloader.download_all_sections()


if __name__ == "__main__":
    main()