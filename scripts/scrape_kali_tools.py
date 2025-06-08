import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re

# Base URL của trang công cụ Kali
BASE_KALI_TOOLS_URL = "https://www.kali.org/tools/"
# File để lưu dữ liệu đã scrape
OUTPUT_DATA_FILE = "data/kali_tools_data.json"
# Thư mục để lưu Chroma DB
CHROMA_DB_DIR = "./chroma_db"
# Thời gian chờ giữa các request để tránh bị chặn IP
REQUEST_DELAY = 1.0 # seconds (Increased for politeness and stability)

# Regex để loại bỏ prompt và các dòng không cần thiết khác
PROMPT_REGEX = re.compile(r"^(root@kali:~# |\$ |.*?@kali:.*# |\$ |# |\(\S+\) $)\s*")
MAN_PAGE_HEADER_FOOTER_REGEX = re.compile(
    r"(^NAME\s+|^SYNOPSIS\s+|^DESCRIPTION\s+|^OPTIONS\s+|^EXAMPLE\s+|^SEE ALSO\s+|^AUTHOR\s+|^<\S+@\S+>\s*)" # Common man page sections
    r"|(\S+\s+General Commands Manual\s+\S+)|(\S+\s+System Manager's Manual\s+\S+)" # Headers/footers like "JOHN(8) System Manager's Manual JOHN(8)"
    r"|(Licensed under AGPL v3.0.*)" # Common license info
    r"|(Copyright \(c\) \d{4}.*?)" # Copyright lines
    r"|(This manual page was written by.*)" # Manual page author lines
    r"|(^.*:\s+invalid option -- '-h'.*)" # Common Python script help messages
    r"|(^Usage: .*--help.*)|(^\s*Options:.*)|(^\s*positional arguments:.*)|(^\s*optional arguments:.*)" # Common help message starts
    r"|(^\s*Find more Information:.*)|(^\s*See doc/.*)" # Other help message info
)

def fetch_page_content(url):
    """Fetches content of a given URL, with basic error handling and headers to mimic a browser."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() 
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def clean_command_output(output):
    """Removes shell prompts, man page headers/footers, and irrelevant lines from command output."""
    lines = output.splitlines()
    cleaned_lines = []
    
    is_man_page_content = False
    for line in lines:
        stripped_line = line.strip()

        if re.search(r"NAME|SYNOPSIS|DESCRIPTION|OPTIONS|EXAMPLE|SEE ALSO|AUTHOR", stripped_line) and len(stripped_line.split()) < 10:
            is_man_page_content = True
        
        if is_man_page_content:
            if stripped_line.startswith("EXAMPLE"):
                cleaned_lines.append(stripped_line)
                is_man_page_content = False
            elif MAN_PAGE_HEADER_FOOTER_REGEX.search(stripped_line):
                continue
            else:
                cleaned_lines.append(stripped_line)
        else:
            if PROMPT_REGEX.match(stripped_line):
                stripped_line = PROMPT_REGEX.sub("", stripped_line, 1)
            if not stripped_line:
                continue
            cleaned_lines.append(stripped_line)

    final_cleaned_lines = []
    for line in cleaned_lines:
        if not line.strip() or re.match(r"^\x1b\[[0-9;]*m$", line.strip()):
            continue
        final_cleaned_lines.append(line)
            
    return "\n".join(final_cleaned_lines).strip()

def scrape_main_kali_tools_page(base_url):
    """
    Scrapes the main Kali tools page to get a list of all individual tool URLs.
    Extracts tool name and its specific URL from the 'card' divs.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Scraping main page: {base_url}")
    html_content = fetch_page_content(base_url)
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    
    tool_cards = soup.find_all('div', class_='card')
    
    found_tool_urls = []
    for card in tool_cards:
        link_tag = card.find('a', href=True)
        if link_tag and 'href' in link_tag.attrs:
            href = link_tag['href']
            full_link_text = link_tag.get_text(strip=True)
            tool_name = full_link_text.split('<span')[0].strip() if '<span' in full_link_text else full_link_text
            tool_name = tool_name.strip() 
            if href.startswith(base_url) and \
               href != base_url and \
               '#missing-tool-banner' not in href and \
               '#all-tools-switch' not in href and \
               'all-tools' not in href: 
                base_tool_url = href.split('#')[0]
                if base_tool_url.endswith('/'):
                     found_tool_urls.append({"name": tool_name, "url": base_tool_url})

    unique_tool_urls_dict = {tool['url']: tool for tool in found_tool_urls}
    
    print(f"[{time.strftime('%H:%M:%S')}] Found {len(unique_tool_urls_dict)} unique tool URLs.")
    return list(unique_tool_urls_dict.values())

def scrape_single_tool_page(tool_info):
    """
    Scrapes an individual tool's page for detailed information including commands.
    """
    tool_name = tool_info['name']
    tool_url = tool_info['url']
    print(f"[{time.strftime('%H:%M:%S')}] Scraping details for '{tool_name}' from {tool_url}")

    html_content = fetch_page_content(tool_url)
    if not html_content:
        return None

    soup = BeautifulSoup(html_content, 'html.parser')

    main_description = ""
    h3_main_tool = soup.find('h3', id=tool_name.lower().replace(' ', '-'))
    if not h3_main_tool:
        h3_main_tool = soup.find('h3', string=lambda text: text and tool_name.lower() in text.lower())

    if h3_main_tool:
        p_tags = []
        current_tag = h3_main_tool.find_next_sibling()
        while current_tag and current_tag.name == 'p':
            p_tags.append(current_tag.get_text(separator=" ", strip=True))
            current_tag = current_tag.find_next_sibling()
        main_description = " ".join(p_tags).strip()
    main_description = main_description or "No detailed description available."


    install_command = ""
    install_info_strong_tag = soup.find('strong', string='How to install:')
    if install_info_strong_tag:
        code_tag = install_info_strong_tag.find_next_sibling('code')
        if code_tag:
            install_command = code_tag.get_text(strip=True)
    install_command = install_command or "Installation command not found (check Kali apt)."


    # Extract specific commands and their usage examples
    commands = []
    # Find all <h5> tags, which often represent sub-commands or specific command usages
    for cmd_heading in soup.find_all('h5'):
        # Get the clean command name from the <h5> tag
        display_cmd_name = cmd_heading.get_text(strip=True)
        # The actual shell example is typically in a <pre><code class="language-console ..."> block
        code_block = cmd_heading.find_next_sibling('pre')
        
        command_usage = ""
        if code_block:
            # Check for the specific class that indicates a shell command output
            code_tag = code_block.find('code', class_='language-console') # Target specific console examples
            if code_tag:
                command_usage = code_tag.get_text(strip=True)
                command_usage = clean_command_output(command_usage)
        
        if display_cmd_name:
            commands.append({
                "sub_command": display_cmd_name,
                "usage_example": command_usage or "No specific usage example provided."
            })
    
    # Also capture commands listed directly in the <a> tag title attribute (e.g., "Includes <command> command")
    main_tool_link = soup.find('a', href=tool_url, recursive=False) # Find the direct link to this tool from its card
    if main_tool_link:
        title_attr = main_tool_link.find('span', title=True)
        if title_attr and 'Includes' in title_attr['title']:
            direct_cmd_name = title_attr['title'].replace('Includes ', '').replace(' command', '').strip()
            # Add this as a primary command if it's not already covered
            if not any(cmd['sub_command'] == direct_cmd_name for cmd in commands):
                commands.insert(0, { # Insert at beginning for primary command
                    "sub_command": direct_cmd_name,
                    "usage_example": f"Use `{direct_cmd_name}` directly. Consult `{direct_cmd_name} -h` or `man {direct_cmd_name}` for options."
                })

    return {
        "name": tool_name,
        "url": tool_url,
        "main_description": main_description,
        "how_to_install": install_command,
        "commands": commands,
        "category": tool_info.get("category", "Unknown") # Keep original category or use a default
    }

def save_data(data, filename):
    """Saves the scraped data to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"\n[{time.strftime('%H:%M:%S')}] Data saved to {filename}")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True) # Ensure Chroma DB directory exists
    
    print("Starting Kali Tools data scraping...")
    
    # 1. Scrape main page to get all tool URLs
    all_tool_urls = scrape_main_kali_tools_page(BASE_KALI_TOOLS_URL)

    # 2. Scrape each individual tool page for details
    detailed_tools_data = []
    total_tools = len(all_tool_urls)
    for i, tool_info in enumerate(all_tool_urls):
        detail = scrape_single_tool_page(tool_info)
        if detail:
            detailed_tools_data.append(detail)
        
        # Be polite, avoid hammering the server
        time.sleep(REQUEST_DELAY)
        
        if (i + 1) % 10 == 0 or (i + 1) == total_tools: # Print progress more frequently
            print(f"[{time.strftime('%H:%M:%S')}] Processed {i+1}/{total_tools} tools...")

    # 3. Save the consolidated data
    save_data(detailed_tools_data, OUTPUT_DATA_FILE)
    print(f"\n[{time.strftime('%H:%M:%S')}] Scraping complete. Scraped {len(detailed_tools_data)} tool details.")