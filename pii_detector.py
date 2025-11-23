"""
OpenPII Watcher: PII Detection Engine
Core detection module with improved regex patterns
"""

import re
from typing import Dict, List, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PIIMatch:
    """Represents a detected PII instance"""
    pii_type: str
    value: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0


class PIIDetector:
    """
    Enhanced PII detector with comprehensive regex patterns
    Supports: emails, phone numbers, names, addresses, SSN, credit cards
    """
    
    def __init__(self):
        # Improved regex patterns based on midterm feedback
        self.patterns = {
            # Email: Standard RFC format + edge cases (plus signs, subdomains)
            'email': r'\b[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,}\b',
            
            # Phone: US formats with various separators including parentheses
            # Handles: (123) 456-7890, 123-456-7890, 123.456.7890, 1234567890
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            
            # Names: Multiple formats
            # First Last, First Middle Last, First M. Last, Dr. First Last
            'name': r'\b(?:(?:Dr|Mr|Ms|Mrs|Prof)\.?\s+)?([A-Z][a-z]+(?:[-\'][A-Z][a-z]+)?)\s+(?:([A-Z]\.?\s+))?([A-Z][a-z]+(?:[-\'][A-Z][a-z]+)?)\b',
            
            # US Street Address: Number + Street + Type
            'address': r'\b\d{1,5}\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir)\b',
            
            # SSN: XXX-XX-XXXX format
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            
            # Credit Card: 16 digits with optional spaces/dashes
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {
            pii_type: re.compile(pattern, re.MULTILINE)
            for pii_type, pattern in self.patterns.items()
        }
        
        # Common false positive filters
        self.false_positive_filters = {
            'phone': self._filter_phone_false_positives,
            'name': self._filter_name_false_positives,
            'credit_card': self._filter_credit_card_false_positives
        }
    
    def _filter_phone_false_positives(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Filter out dates and other number sequences mistaken for phones"""
        filtered = []
        for match in matches:
            # Remove dates like 12-31-2024
            digits = re.findall(r'\d+', match.value)
            if len(digits) == 3:
                # Check if it looks like a date
                if int(digits[0]) <= 12 and int(digits[1]) <= 31:
                    continue
            filtered.append(match)
        return filtered
    
    def _filter_name_false_positives(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Filter out common words mistaken for names"""
        common_words = {'The', 'This', 'That', 'Will', 'May', 'June', 'July', 
                       'August', 'March', 'April', 'January', 'February'}
        filtered = []
        for match in matches:
            words = match.value.split()
            if not any(word in common_words for word in words):
                filtered.append(match)
        return filtered
    
    def _filter_credit_card_false_positives(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Basic Luhn algorithm check for credit card validation"""
        filtered = []
        for match in matches:
            digits = re.sub(r'[-\s]', '', match.value)
            if self._luhn_check(digits):
                filtered.append(match)
        return filtered
    
    def _luhn_check(self, card_number: str) -> bool:
        """Luhn algorithm for credit card validation"""
        def digits_of(n):
            return [int(d) for d in str(n)]
        
        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        return checksum % 10 == 0
    
    def detect(self, text: str, pii_type: str) -> List[PIIMatch]:
        """Detect specific PII type in text"""
        if pii_type not in self.compiled_patterns:
            raise ValueError(f"Unknown PII type: {pii_type}")
        
        pattern = self.compiled_patterns[pii_type]
        matches = []
        
        for match in pattern.finditer(text):
            pii_match = PIIMatch(
                pii_type=pii_type,
                value=match.group(0),
                start_pos=match.start(),
                end_pos=match.end()
            )
            matches.append(pii_match)
        
        # Apply filters if available
        if pii_type in self.false_positive_filters:
            matches = self.false_positive_filters[pii_type](matches)
        
        return matches
    
    def detect_all(self, text: str) -> Dict[str, List[PIIMatch]]:
        """Detect all PII types in text"""
        results = {}
        for pii_type in self.patterns.keys():
            results[pii_type] = self.detect(text, pii_type)
        return results
    
    def get_summary(self, text: str) -> Dict[str, int]:
        """Get summary counts of detected PII"""
        all_results = self.detect_all(text)
        return {
            pii_type: len(matches)
            for pii_type, matches in all_results.items()
        }
    
    def get_unique_values(self, text: str) -> Dict[str, Set[str]]:
        """Get unique PII values detected"""
        all_results = self.detect_all(text)
        return {
            pii_type: set(match.value for match in matches)
            for pii_type, matches in all_results.items()
        }


if __name__ == "__main__":
    # Test the detector
    detector = PIIDetector()
    
    test_text = """
    Contact Information:
    - John Smith: john.smith@example.com, (555) 123-4567
    - Dr. Mary-Anne Johnson: mary.johnson+work@company.com, 555-987-6543
    - Robert O'Brien: robrien@test.org, 1234567890
    
    Address: 123 Main Street, Apartment 4B
    SSN: 123-45-6789
    Credit Card: 4532-1488-0343-6467
    """
    
    results = detector.detect_all(test_text)
    
    print("=== PII Detection Results ===")
    for pii_type, matches in results.items():
        if matches:
            print(f"\n{pii_type.upper()}: {len(matches)} found")
            for match in matches:
                print(f"  - {match.value} (pos: {match.start_pos}-{match.end_pos})")

