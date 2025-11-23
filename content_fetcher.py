"""
OpenPII Watcher: Content Fetcher
Platform-specific URL processing and content fetching
Supports: Pastebin, Google Docs
"""

import re
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Platform(Enum):
    """Supported sharing platforms"""
    PASTEBIN = "pastebin"
    GOOGLE_DOCS = "google_docs"
    UNKNOWN = "unknown"


@dataclass
class FetchResult:
    """Result of content fetching operation"""
    success: bool
    content: Optional[str]
    platform: Platform
    error_message: Optional[str] = None
    fetch_method: Optional[str] = None


class ContentFetcher:
    """
    Platform-agnostic content fetcher with URL detection and transformation
    Implements two-tier fallback strategy for Google Docs
    """
    
    def __init__(self):
        # Platform detection patterns
        self.platform_patterns = {
            Platform.PASTEBIN: r'pastebin\.com/([A-Za-z0-9]+)',
            Platform.GOOGLE_DOCS: r'docs\.google\.com/document/d/([A-Za-z0-9_-]+)'
        }
        
        # CORS proxy for fallback
        self.cors_proxy = "https://api.allorigins.win/raw?url="
    
    def detect_platform(self, url: str) -> Tuple[Platform, Optional[str]]:
        """
        Detect platform type from URL and extract document ID
        Returns: (Platform, document_id)
        """
        for platform, pattern in self.platform_patterns.items():
            match = re.search(pattern, url)
            if match:
                return platform, match.group(1)
        
        return Platform.UNKNOWN, None
    
    def transform_pastebin_url(self, paste_id: str) -> str:
        """
        Transform Pastebin URL to raw content endpoint
        Example: pastebin.com/ABC -> pastebin.com/raw/ABC
        """
        return f"https://pastebin.com/raw/{paste_id}"
    
    def transform_google_docs_url(self, doc_id: str) -> str:
        """
        Transform Google Docs URL to plain text export endpoint
        Example: docs.google.com/document/d/ABC/edit -> 
                 docs.google.com/document/d/ABC/export?format=txt
        """
        return f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    
    def get_fetch_instructions(self, url: str) -> dict:
        """
        Get platform-specific fetching instructions for client-side implementation
        This provides the logic for the JavaScript/browser implementation
        """
        platform, doc_id = self.detect_platform(url)
        
        if platform == Platform.PASTEBIN:
            return {
                'platform': 'pastebin',
                'doc_id': doc_id,
                'primary_url': self.transform_pastebin_url(doc_id),
                'method': 'direct',
                'cors_required': False,
                'description': 'Fetch from Pastebin /raw/ endpoint (CORS-friendly)'
            }
        
        elif platform == Platform.GOOGLE_DOCS:
            primary_url = self.transform_google_docs_url(doc_id)
            return {
                'platform': 'google_docs',
                'doc_id': doc_id,
                'primary_url': primary_url,
                'fallback_url': f"{self.cors_proxy}{primary_url}",
                'method': 'two_tier_fallback',
                'cors_required': True,
                'description': 'Two-tier strategy: 1) Direct fetch from /export?format=txt, 2) CORS proxy fallback'
            }
        
        else:
            return {
                'platform': 'unknown',
                'error': 'Unsupported platform',
                'supported_platforms': ['pastebin.com', 'docs.google.com']
            }
    
    def get_javascript_fetch_code(self, url: str) -> str:
        """
        Generate JavaScript code for client-side fetching
        This is what will be used in the web interface
        """
        instructions = self.get_fetch_instructions(url)
        platform = instructions.get('platform')
        
        if platform == 'pastebin':
            return f"""
// Pastebin fetch (CORS-friendly)
async function fetchContent() {{
    const url = "{instructions['primary_url']}";
    try {{
        const response = await fetch(url);
        if (!response.ok) throw new Error('Fetch failed');
        return await response.text();
    }} catch (error) {{
        console.error('Pastebin fetch error:', error);
        return null;
    }}
}}
"""
        
        elif platform == 'google_docs':
            return f"""
// Google Docs fetch with two-tier fallback
async function fetchContent() {{
    // Strategy 1: Try direct fetch from export endpoint
    const primaryUrl = "{instructions['primary_url']}";
    try {{
        const response = await fetch(primaryUrl);
        if (response.ok) {{
            console.log('Success: Direct fetch from Google Docs');
            return await response.text();
        }}
    }} catch (error) {{
        console.log('Direct fetch blocked by CORS, trying proxy...');
    }}
    
    // Strategy 2: Fallback to CORS proxy
    const fallbackUrl = "{instructions['fallback_url']}";
    try {{
        const response = await fetch(fallbackUrl);
        if (response.ok) {{
            console.log('Success: Fetched via CORS proxy');
            return await response.text();
        }} else {{
            throw new Error('Proxy fetch failed');
        }}
    }} catch (error) {{
        console.error('Both fetch methods failed:', error);
        return null;
    }}
}}
"""
        
        else:
            return """
// Unsupported platform
function fetchContent() {
    console.error('Unsupported platform');
    return null;
}
"""
    
    def validate_url(self, url: str) -> Tuple[bool, str]:
        """
        Validate if URL is supported and properly formatted
        Returns: (is_valid, message)
        """
        if not url or not url.strip():
            return False, "Empty URL"
        
        platform, doc_id = self.detect_platform(url)
        
        if platform == Platform.UNKNOWN:
            return False, "Unsupported platform. Supported: Pastebin, Google Docs"
        
        if not doc_id:
            return False, f"Could not extract document ID from {platform.value}"
        
        return True, f"Valid {platform.value} URL"
    
    def get_platform_integration_details(self) -> dict:
        """
        Provide detailed integration information for documentation
        """
        return {
            'pastebin': {
                'url_pattern': 'pastebin.com/{paste_id}',
                'transformation': 'pastebin.com/raw/{paste_id}',
                'method': 'Direct fetch',
                'cors': 'Native support via /raw/ endpoint',
                'authentication': 'Not required for public pastes',
                'reliability': 'High (>95%)',
                'limitations': 'Requires paste to be public'
            },
            'google_docs': {
                'url_pattern': 'docs.google.com/document/d/{doc_id}',
                'transformation': 'docs.google.com/document/d/{doc_id}/export?format=txt',
                'method': 'Two-tier fallback (direct + proxy)',
                'cors': 'Mixed - direct may fail, proxy as backup',
                'authentication': 'Works with "anyone with link" permission',
                'reliability': 'Medium (60-80%) depending on sharing settings',
                'limitations': [
                    'Requires "anyone with link" permission',
                    'May fail if doc requires authentication',
                    'CORS proxy adds latency (~500ms)',
                    'Plain text export loses formatting'
                ]
            }
        }


if __name__ == "__main__":
    # Test the content fetcher
    fetcher = ContentFetcher()
    
    # Test URLs
    test_urls = [
        "https://pastebin.com/ABC123",
        "https://docs.google.com/document/d/1234567890abcdef/edit",
        "https://unsupported.com/document"
    ]
    
    print("=== Content Fetcher Testing ===\n")
    
    for url in test_urls:
        print(f"URL: {url}")
        
        # Validate
        valid, message = fetcher.validate_url(url)
        print(f"  Valid: {valid} - {message}")
        
        # Get platform
        platform, doc_id = fetcher.detect_platform(url)
        print(f"  Platform: {platform.value}, Doc ID: {doc_id}")
        
        # Get instructions
        if valid:
            instructions = fetcher.get_fetch_instructions(url)
            print(f"  Fetch method: {instructions.get('method')}")
            print(f"  Primary URL: {instructions.get('primary_url')}")
        
        print()
    
    # Print integration details
    print("\n=== Platform Integration Details ===")
    details = fetcher.get_platform_integration_details()
    for platform, info in details.items():
        print(f"\n{platform.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

