"""
OpenPII Watcher: Synthetic Test Data Generator
Generates realistic test documents with ground truth labels
"""

import random
from typing import List, Dict, Set
from evaluator import GroundTruth
import json


class TestDataGenerator:
    """
    Generate synthetic documents with known PII for testing
    Creates diverse scenarios: contact lists, forms, mixed documents
    """
    
    def __init__(self, seed=42):
        random.seed(seed)
        
        # Sample data pools
        self.first_names = [
            'John', 'Mary', 'Robert', 'Jennifer', 'Michael', 'Linda',
            'David', 'Sarah', 'James', 'Emily', 'William', 'Jessica',
            'Richard', 'Patricia', 'Joseph', 'Nancy', 'Thomas', 'Lisa',
            'Dr. Elizabeth', 'Prof. Andrew', "Mary-Anne", "O'Brien"
        ]
        
        self.last_names = [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia',
            'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez',
            'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor',
            'Moore', 'Jackson', 'Martin', "O'Connor", 'Thompson'
        ]
        
        self.domains = [
            'example.com', 'test.com', 'company.com', 'email.com',
            'work.org', 'university.edu', 'business.net', 'mail.com'
        ]
        
        self.street_names = [
            'Main', 'Oak', 'Pine', 'Maple', 'Cedar', 'Elm',
            'Washington', 'Lake', 'Hill', 'Park', 'First', 'Second'
        ]
        
        self.street_types = ['Street', 'St', 'Avenue', 'Ave', 'Road', 'Rd', 'Boulevard', 'Blvd', 'Lane', 'Ln', 'Drive', 'Dr']
    
    def generate_name(self) -> str:
        """Generate random name"""
        first = random.choice(self.first_names)
        last = random.choice(self.last_names)
        
        # Sometimes add middle initial
        if random.random() < 0.3:
            middle = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            return f"{first} {middle}. {last}"
        
        return f"{first} {last}"
    
    def generate_email(self, name: str = None) -> str:
        """Generate email, optionally based on name"""
        if name and random.random() < 0.7:
            # Generate email from name
            parts = name.replace('Dr.', '').replace('Prof.', '').replace('.', '').strip().split()
            if len(parts) >= 2:
                first = parts[0].lower()
                last = parts[-1].lower().replace("'", '').replace('-', '')
                
                # Various email formats
                formats = [
                    f"{first}.{last}",
                    f"{first}{last}",
                    f"{first[0]}{last}",
                    f"{first}_{last}",
                    f"{first}+work"  # Edge case with plus sign
                ]
                username = random.choice(formats)
            else:
                username = f"user{random.randint(100, 999)}"
        else:
            username = f"user{random.randint(100, 999)}"
        
        domain = random.choice(self.domains)
        return f"{username}@{domain}"
    
    def generate_phone(self) -> str:
        """Generate US phone number in various formats"""
        area = random.randint(200, 999)
        prefix = random.randint(200, 999)
        line = random.randint(1000, 9999)
        
        formats = [
            f"({area}) {prefix}-{line}",
            f"{area}-{prefix}-{line}",
            f"{area}.{prefix}.{line}",
            f"{area}{prefix}{line}"
        ]
        
        return random.choice(formats)
    
    def generate_address(self) -> str:
        """Generate street address"""
        number = random.randint(1, 9999)
        street = random.choice(self.street_names)
        stype = random.choice(self.street_types)
        
        return f"{number} {street} {stype}"
    
    def generate_ssn(self) -> str:
        """Generate fake SSN"""
        return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
    
    def generate_credit_card(self) -> str:
        """Generate fake credit card (Luhn-valid)"""
        # Simple approach: use known test card numbers
        test_cards = [
            "4532-1488-0343-6467",  # Visa
            "5425-2334-3010-9903",  # Mastercard
            "3782-822463-10005",    # Amex
        ]
        return random.choice(test_cards)
    
    def generate_contact_list(self, num_contacts: int = 5) -> GroundTruth:
        """Generate a contact list document"""
        lines = ["CONTACT LIST", "=" * 50, ""]
        
        labels = {
            'name': set(),
            'email': set(),
            'phone': set(),
            'address': set(),
            'ssn': set(),
            'credit_card': set()
        }
        
        for i in range(num_contacts):
            name = self.generate_name()
            email = self.generate_email(name)
            phone = self.generate_phone()
            
            labels['name'].add(name)
            labels['email'].add(email)
            labels['phone'].add(phone)
            
            lines.append(f"{i+1}. {name}")
            lines.append(f"   Email: {email}")
            lines.append(f"   Phone: {phone}")
            lines.append("")
        
        text = "\n".join(lines)
        return GroundTruth(
            sample_id=f"contact_list_{num_contacts}",
            text=text,
            labels=labels
        )
    
    def generate_signup_sheet(self, num_signups: int = 8) -> GroundTruth:
        """Generate event signup sheet"""
        lines = ["EVENT SIGNUP SHEET", "Please provide your contact information", "=" * 60, ""]
        
        labels = {
            'name': set(),
            'email': set(),
            'phone': set(),
            'address': set(),
            'ssn': set(),
            'credit_card': set()
        }
        
        for i in range(num_signups):
            name = self.generate_name()
            email = self.generate_email(name)
            phone = self.generate_phone()
            
            labels['name'].add(name)
            labels['email'].add(email)
            labels['phone'].add(phone)
            
            lines.append(f"Name: {name}")
            lines.append(f"Email: {email}, Phone: {phone}")
            lines.append("-" * 60)
        
        text = "\n".join(lines)
        return GroundTruth(
            sample_id=f"signup_sheet_{num_signups}",
            text=text,
            labels=labels
        )
    
    def generate_mixed_document(self) -> GroundTruth:
        """Generate document with mixed content and various PII"""
        name1 = self.generate_name()
        name2 = self.generate_name()
        email1 = self.generate_email(name1)
        email2 = self.generate_email(name2)
        phone1 = self.generate_phone()
        phone2 = self.generate_phone()
        address1 = self.generate_address()
        ssn1 = self.generate_ssn()
        card1 = self.generate_credit_card()
        
        text = f"""
Meeting Notes - Project Planning
Date: March 15, 2024

Attendees:
- {name1} ({email1})
- {name2} ({email2})

Contact Information:
{name1}: {phone1}
{name2}: {phone2}

Office Location: {address1}

Financial Information (CONFIDENTIAL):
SSN: {ssn1}
Payment Card: {card1}

Additional Notes:
- Follow up with team members next week
- Budget approval pending
- Contact information should be kept secure

Some random numbers that are NOT phone numbers:
- Order ID: 123-456-7890 (actually this looks like a phone!)
- Date: 12-31-2024
- Reference: 987-654-3210
"""
        
        labels = {
            'name': {name1, name2},
            'email': {email1, email2},
            'phone': {phone1, phone2},
            'address': {address1},
            'ssn': {ssn1},
            'credit_card': {card1}
        }
        
        return GroundTruth(
            sample_id="mixed_document",
            text=text,
            labels=labels
        )
    
    def generate_edge_cases(self) -> GroundTruth:
        """Generate document with edge cases for testing"""
        text = """
Edge Case Testing Document

1. Emails with special characters:
   - john+work@example.com
   - user.name+tag@company.com
   - firstname_lastname@test.org

2. Names with special formatting:
   - Mary-Anne Johnson (hyphenated first name)
   - Dr. Robert O'Brien (prefix and apostrophe)
   - Lisa Smith-Anderson (hyphenated last name)

3. Phone numbers without standard formatting:
   - 5551234567 (no separators)
   - (555) 987-6543 (parentheses and dash)

4. False positives to avoid:
   - Date: 12-31-2024 (not a phone number)
   - ID: 123-456-7890 (could be phone or ID)
   - Numbers: 111-222-3333 (sequential, suspicious)

5. International scenarios:
   - Non-US name: Rajesh Kumar Patel
   - Email with subdomain: user@mail.company.com
"""
        
        labels = {
            'name': {'Mary-Anne Johnson', 'Dr. Robert O\'Brien', 'Lisa Smith-Anderson', 'Rajesh Kumar Patel'},
            'email': {'john+work@example.com', 'user.name+tag@company.com', 'firstname_lastname@test.org', 'user@mail.company.com'},
            'phone': {'5551234567', '(555) 987-6543'},
            'address': set(),
            'ssn': set(),
            'credit_card': set()
        }
        
        return GroundTruth(
            sample_id="edge_cases",
            text=text,
            labels=labels
        )
    
    def generate_test_dataset(self, num_samples: int = 20) -> List[GroundTruth]:
        """Generate comprehensive test dataset"""
        dataset = []
        
        # Generate different types of documents
        for i in range(num_samples // 4):
            dataset.append(self.generate_contact_list(random.randint(3, 8)))
            dataset.append(self.generate_signup_sheet(random.randint(5, 10)))
            dataset.append(self.generate_mixed_document())
        
        # Add edge cases
        dataset.append(self.generate_edge_cases())
        
        return dataset
    
    def save_dataset(self, dataset: List[GroundTruth], filename: str):
        """Save dataset to JSON file"""
        data = []
        for gt in dataset:
            data.append({
                'sample_id': gt.sample_id,
                'text': gt.text,
                'labels': {k: list(v) for k, v in gt.labels.items()}
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Dataset saved to {filename}")
    
    def load_dataset(self, filename: str) -> List[GroundTruth]:
        """Load dataset from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        dataset = []
        for item in data:
            gt = GroundTruth(
                sample_id=item['sample_id'],
                text=item['text'],
                labels={k: set(v) for k, v in item['labels'].items()}
            )
            dataset.append(gt)
        
        return dataset


if __name__ == "__main__":
    # Generate test dataset
    generator = TestDataGenerator()
    
    print("=== Generating Test Dataset ===\n")
    
    # Generate sample documents
    contact_list = generator.generate_contact_list(3)
    print("Sample Contact List:")
    print(contact_list.text[:200] + "...")
    print(f"\nGround truth labels:")
    for pii_type, values in contact_list.labels.items():
        if values:
            print(f"  {pii_type}: {len(values)} items")
    
    print("\n" + "=" * 60 + "\n")
    
    # Generate full dataset
    dataset = generator.generate_test_dataset(20)
    print(f"Generated {len(dataset)} test samples")
    
    # Save to file
    generator.save_dataset(dataset, 'test_dataset.json')

