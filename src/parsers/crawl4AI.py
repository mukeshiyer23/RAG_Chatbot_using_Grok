import asyncio
import aiohttp
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Set, List, Dict
import os
import re
import chardet
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class PageInfo:
    url: str
    has_table: bool = False
    has_pagination: bool = False
    document_links: List[Dict[str, str]] = None


class SiteMapperBot:
    def __init__(self, base_url: str, output_dir: str = "site_content"):
        self.base_url = base_url.rstrip('/')
        self.output_dir = output_dir
        self.docs_dir = os.path.join(output_dir, "documents")
        self.mapping_dir = os.path.join(output_dir, "mapping")
        self.processed_urls = []
        # Track visited URLs and discovered documents
        self.visited_urls: Set[str] = set()
        self.discovered_pages: List[PageInfo] = []
        self.downloaded_docs: Set[str] = set()
        self.markdown_urls: Set[str] = set()

        # Create directories
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.mapping_dir, exist_ok=True)
        os.makedirs(os.path.join(self.docs_dir, "markdown"), exist_ok=True)

        # Initialize session
        self.session = None

    def safe_filename(self, filename: str) -> str:
        """Create a safe filename by removing invalid characters"""
        return re.sub(r'[<>:"/\\|?*]', '_', filename)

    def detect_encoding(self, content: bytes) -> str:
        """Detect file encoding with more robust fallback"""
        try:
            result = chardet.detect(content)
            detected_encoding = result.get('encoding', 'utf-8')

            # Fallback to utf-8 with error handling for problematic encodings
            return detected_encoding if detected_encoding else 'utf-8'
        except Exception:
            return 'utf-8'

    def clean_url(self, url: str) -> str:
        """Clean and normalize URL"""
        if not url or url.startswith('#'):
            return ''

        # Create absolute URL
        full_url = urljoin(self.base_url, url)

        # Remove fragments
        parsed = urlparse(full_url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def extract_sitemap_urls(self, sitemap_content: str) -> List[str]:
        """Extract URLs from sitemap.xml"""
        try:
            root = ET.fromstring(sitemap_content)

            # Handle both standard sitemap and sitemap index formats
            urls = []
            for elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                url = elem.text.strip()
                cleaned_url = self.clean_url(url)

                # Exclude undesired URLs
                if (cleaned_url and
                        cleaned_url not in self.visited_urls):
                    urls.append(cleaned_url)

            return list(set(urls))
        except Exception as e:
            print(f"Error parsing sitemap: {e}")
            return []

    async def extract_page_content(self, url: str) -> str:
        """Extract main text content from a page with robust encoding handling"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    print(f"Failed to fetch {url}: {response.status}")
                    return ""

                # Use better encoding detection and decoding
                content = await response.read()

                # Try multiple encoding strategies
                encodings_to_try = ['utf-8', 'iso-8859-1', 'latin1']

                for encoding in encodings_to_try:
                    try:
                        html_content = content.decode(encoding, errors='ignore')
                        soup = BeautifulSoup(html_content, 'html.parser')
                        break
                    except Exception:
                        continue
                else:
                    print(f"Failed to decode content from {url}")
                    return ""

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Extract text from main content areas
                text_content = ""
                main_content_tags = ['article', 'main', 'div.content', '#content', '.content']

                for tag in main_content_tags:
                    content_area = soup.select_one(tag)
                    if content_area:
                        text_content = content_area.get_text(strip=True, separator='\n')
                        break

                # Fallback to body text if no specific content area found
                if not text_content:
                    text_content = soup.get_text(strip=True, separator='\n')

                return text_content
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return ""

    async def download_as_markdown(self, url: str) -> bool:
        """
        Universal markdown extraction with intelligent parsing

        Args:
            url (str): Source URL to extract

        Returns:
            bool: Download success status
        """
        try:
            # Prevent duplicate downloads
            if url in self.processed_urls:
                return False

            # Extract raw content
            content = await self.extract_page_content(url)

            if not content:
                print(f"No content extracted from {url}")
                return False

            # Generate markdown filename
            filename = self.safe_filename(
                os.path.basename(urlparse(url).path) or 'page'
            )
            markdown_filename = f"{filename}.md"
            markdown_dir = os.path.join(self.docs_dir, 'markdown')
            filepath = os.path.join(markdown_dir, markdown_filename)

            # Smart content conversion
            markdown_content = self.convert_to_markdown(content, url)

            # Write markdown with enhanced metadata
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"---\n")
                f.write(f"source_url: {url}\n")
                f.write(f"extracted_at: {datetime.now().isoformat()}\n")
                f.write("---\n\n")
                f.write(markdown_content)

            print(f"Extracted markdown: {markdown_filename}")
            self.processed_urls.append(url)
            return True

        except Exception as e:
            print(f"Markdown extraction error for {url}: {e}")
            return False

    def convert_to_markdown(self, content: str, source_url: str) -> str:
        """
        Intelligent markdown conversion

        Args:
            content (str): Raw content
            source_url (str): Original source URL

        Returns:
            str: Converted markdown content
        """
        # HTML parsing
        if '<html' in content or '<!DOCTYPE' in content:
            return self.html_to_markdown(content)

        # JSON/XML parsing
        if content.lstrip().startswith('{') or content.lstrip().startswith('<'):
            return self.structured_data_to_markdown(content)

        # Plain text fallback
        return self.plain_text_to_markdown(content)

    def html_to_markdown(self, html_content: str) -> str:
        """Convert HTML to clean markdown"""
        try:
            # Use html2text or similar library
            import html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            return h.handle(html_content)
        except Exception:
            # Fallback parsing
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(separator='\n\n')

    def structured_data_to_markdown(self, data: str) -> str:
        """Convert JSON/XML to markdown table or list"""
        try:
            import json
            import xmltodict

            # Attempt JSON first
            try:
                parsed_data = json.loads(data)
                return self.json_to_markdown(parsed_data)
            except json.JSONDecodeError:
                # Fallback to XML
                parsed_data = xmltodict.parse(data)
                return self.json_to_markdown(parsed_data)
        except Exception as e:
            return f"## Parsing Error\n\n{str(e)}"

    def json_to_markdown(self, data: dict, indent: int = 0) -> str:
        """Recursive JSON to markdown conversion"""
        markdown = ""
        indent_str = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    markdown += f"{indent_str}### {key}\n\n"
                    markdown += self.json_to_markdown(value, indent + 1)
                else:
                    markdown += f"{indent_str}- **{key}**: {value}\n"

        elif isinstance(data, list):
            for item in data:
                markdown += self.json_to_markdown(item, indent)

        return markdown

    def plain_text_to_markdown(self, text: str) -> str:
        """Convert plain text to markdown"""
        # Basic text structuring
        lines = text.split('\n')
        markdown_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect potential headings
            if len(line) > 40 and line.isupper():
                markdown_lines.append(f"## {line}")
            else:
                markdown_lines.append(line)

        return '\n\n'.join(markdown_lines)

    def has_pagination(self, soup: BeautifulSoup) -> bool:
        """Check if page has pagination"""
        pagination_selectors = [
            '.dataTables_paginate',
            '.pagination',
            '.pager',
            'nav.navigation',
            '[class*="page-numbers"]'
        ]

        return any(bool(soup.select(selector)) for selector in pagination_selectors)

    def has_document_table(self, soup: BeautifulSoup) -> bool:
        """Check if page has document table"""
        table_selectors = [
            'table#GridList',
            'table.dataTable',
            'table.document-list',
            'table.files-list'
        ]

        return any(bool(soup.select(selector)) for selector in table_selectors)

    def extract_document_links(self, soup: BeautifulSoup, page_url: str) -> List[Dict[str, str]]:
        """Extract document links from page"""
        documents = []

        # Expanded document extensions to include markdown
        doc_extensions = ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.md', '.markdown', '.txt')

        # Process table-based documents
        if self.has_document_table(soup):
            for row in soup.select('table#GridList tbody tr'):
                try:
                    date_cell = row.select_one('td:first-child')
                    title_cell = row.select_one('td:nth-child(2)')
                    download_cell = row.select_one('td:last-child a')

                    if download_cell and title_cell:
                        doc_url = self.clean_url(download_cell.get('href', ''))
                        if doc_url and doc_url.lower().endswith(doc_extensions):
                            documents.append({
                                'url': doc_url,
                                'title': title_cell.text.strip(),
                                'date': date_cell.text.strip() if date_cell else '',
                                'type': 'table_document'
                            })
                except Exception as e:
                    print(f"Error extracting table document: {e}")

        # Process regular document links
        for link in soup.find_all('a'):
            try:
                href = link.get('href', '')
                if href.lower().endswith(doc_extensions):
                    doc_url = self.clean_url(href)
                    if doc_url:
                        documents.append({
                            'url': doc_url,
                            'title': link.text.strip() or os.path.basename(doc_url),
                            'date': '',
                            'type': 'inline_document'
                        })
            except Exception as e:
                print(f"Error extracting inline document: {e}")

        return documents

    async def download_document(self, doc_info: Dict[str, str]) -> bool:
        """Download document with comprehensive logging"""
        if doc_info['url'] in self.downloaded_docs:
            print(f"[SKIP] Already downloaded: {doc_info['url']}")
            return False

        try:
            print(f"[DOWNLOAD_ATTEMPT] URL: {doc_info['url']}")
            print(f"[DOC_INFO] {doc_info}")

            response = requests.get(doc_info["url"])
            print(f"[RESPONSE_STATUS] {response.status_code}")
            print(f"[RESPONSE_HEADERS] {dict(response.headers)}")
            if response.status_code == 200:

                # Raw content retrieval
                content = response.content
                print(f"[CONTENT_LENGTH] {len(content)} bytes")

                # Determine file extension and storage directory
                file_ext = os.path.splitext(doc_info['url'])[1].lower() or '.bin'
                doc_dir = os.path.join(self.docs_dir, file_ext.replace('.', '') or 'misc')
                os.makedirs(doc_dir, exist_ok=True)

                # Generate unique filename
                base_filename = self.safe_filename(doc_info.get('title', '') or os.path.basename(doc_info['url']))
                timestamp = datetime.now()
                timestamp_str = timestamp.strftime("%Y%m%d%H%M%S")

                filename = f"IFSCA{timestamp_str}{file_ext}"
                filepath = os.path.join(doc_dir, filename)

                print(f"[FILEPATH] {filepath}")

                # Try decoding first
                try:
                    with open(filepath, "wb") as file:
                        file.write(response.content)
                    print(f"[WRITE_METHOD] Decoded text write")
                except Exception as decode_err:
                    print(f"[DECODE_ERROR] {decode_err}")
                    # Fallback to binary write
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    print(f"[WRITE_METHOD] Binary write")

                # Verify file creation and size
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    print(f"[FILE_CREATED] Size: {file_size} bytes")
                else:
                    print(f"[FILE_CREATE_FAILURE] Could not create file")

                # Save metadata
                metadata = {
                    'title': doc_info.get('title', 'Unknown'),
                    'source_url': doc_info['url'],
                    'downloaded_at': datetime.now().isoformat(),
                    'detected_encoding': response.headers['Content-Type'],
                    'content_length': len(content),
                    'file_extension': file_ext
                }

                meta_filepath = f"{filepath}.meta.json"
                with open(meta_filepath, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                print(f"[METADATA_SAVED] {meta_filepath}")

                self.downloaded_docs.add(doc_info['url'])
                return True

        except Exception as e:
            print(f"[DOWNLOAD_ERROR] {doc_info['url']}")
            print(f"[ERROR_DETAILS] {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

        return False

    async def process_page(self, url: str) -> List[str]:
        """Process a single page and extract information"""
        if url in self.visited_urls or "https://ifsca.gov.in" not in url:
            return []

        self.visited_urls.add(url)
        discovered_links = []

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    print(f"Failed to fetch {url}: {response.status}")
                    return []

                # Use robust decoding
                content = await response.read()

                # Try multiple encoding strategies
                encodings_to_try = ['utf-8', 'iso-8859-1', 'latin1']

                for encoding in encodings_to_try:
                    try:
                        html_content = content.decode(encoding, errors='ignore')
                        soup = BeautifulSoup(html_content, 'html.parser')
                        break
                    except Exception:
                        continue
                else:
                    print(f"Failed to decode content from {url}")
                    return []

                # Extract page information
                has_table = self.has_document_table(soup)
                has_pagination = self.has_pagination(soup)
                document_links = self.extract_document_links(soup, url)

                # Store page info
                page_info = PageInfo(
                    url=url,
                    has_table=has_table,
                    has_pagination=has_pagination,
                    document_links=document_links
                )
                self.discovered_pages.append(page_info)

                # Download documents
                for doc_info in document_links:
                    await self.download_document(doc_info)

                # Extract more links from the page
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href:
                        clean_url = self.clean_url(href)
                        if clean_url and clean_url not in self.visited_urls:
                            discovered_links.append(clean_url)

                print(f"\nProcessed: {url}")
                print(f"Has table: {has_table}")
                print(f"Has pagination: {has_pagination}")
                print(f"Documents found: {len(document_links)}")

        except Exception as e:
            print(f"Error processing {url}: {e}")

        return discovered_links

    async def map_site(self, sitemap_path: str):
        """Map the entire site using sitemap.xml"""
        print(f"\n=== Starting Site Mapping from {sitemap_path} ===\n")

        async with aiohttp.ClientSession() as session:
            self.session = session

            # Read sitemap.xml
            with open(sitemap_path, 'r', encoding='utf-8') as f:
                sitemap_content = f.read()

            # Extract URLs from sitemap
            urls_to_visit = self.extract_sitemap_urls(sitemap_content)

            # Phase 1: Extract Markdown content from all pages
            markdown_tasks = [self.download_as_markdown(url) for url in urls_to_visit]
            await asyncio.gather(*markdown_tasks)

            while urls_to_visit:
                url = urls_to_visit.pop(0)
                if url not in self.visited_urls:
                    new_links = await self.process_page(url)
                    urls_to_visit.extend(new_links)

        # Save mapping results
        mapping_file = os.path.join(self.mapping_dir, 'site_mapping.json')
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump([{
                'url': page.url,
                'has_table': page.has_table,
                'has_pagination': page.has_pagination,
                'document_count': len(page.document_links)
            } for page in self.discovered_pages], f, indent=2)

        print(f"\n=== Site Mapping Complete ===")
        print(f"Pages processed: {len(self.visited_urls)}")
        print(f"Documents downloaded: {len(self.downloaded_docs)}")
        print(f"Mapping saved to: {mapping_file}")


async def main():
    base_url = "https://ifsca.gov.in"
    sitemap_path = "sitemap.xml"

    mapper = SiteMapperBot(base_url)
    await mapper.map_site(sitemap_path)


if __name__ == "__main__":
    asyncio.run(main())