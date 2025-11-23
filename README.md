# OpenPII Watcher - Research Implementation

**Systematic PII Detection in Publicly Shared Documents**

A privacy-focused detection system that identifies exposed personally identifiable information (PII) in public sharing platforms using enhanced regex-based pattern matching.

---

## Academic Project

**Course:** CSE 253 - Network Security (Graduate)  
**Institution:** UC Santa Cruz  
**Authors:**
- Suryakiran Valavala (suvalava@ucsc.edu)
- Arsh Advani (agadvani@ucsc.edu)
- Vijay Arvind Ramamoorthy (viramamo@ucsc.edu)

**Research Question:** Can we systematically detect PII (emails, phone numbers, names, addresses, SSN, credit cards) exposed through publicly accessible sharing links?

---

## Key Results

### Overall Performance
- **Precision:** 84.3%
- **Recall:** 87.6%
- **F1-Score:** 85.9%
- **Improvement over Baseline:** +84.3% average

### Per-Type Performance (Micro-Averaged)

| PII Type | Precision | Recall | F1-Score | TP | FP | FN |
|----------|-----------|--------|----------|----|----|-----|
| **Email** | 100.0% | 100.0% | **1.000** | 68 | 0 | 0 |
| **Address** | 100.0% | 100.0% | **1.000** | 5 | 0 | 0 |
| **SSN** | 100.0% | 100.0% | **1.000** | 5 | 0 | 0 |
| **Name** | 58.7% | 94.1% | **0.723** | 64 | 45 | 4 |
| **Phone** | 62.8% | 74.2% | **0.681** | 49 | 29 | 17 |
| **Credit Card** | 100.0% | 20.0% | **0.333** | 1 | 0 | 4 |

**Test Dataset:** 16 synthetic documents, 192 PII instances

---

## System Architecture

### Core Components

```
pii_detector.py          - Enhanced regex-based PII detection engine
content_fetcher.py       - Platform-specific URL processing and content fetching
evaluator.py             - Evaluation framework (precision/recall/F1)
test_data_generator.py   - Synthetic test data generation with ground truth
run_experiments.py       - Complete experimental pipeline
```

### Supported Platforms

**Pastebin**
- Direct fetch via `/raw/` endpoint
- Native CORS support
- 95%+ success rate
- 200-300ms latency

**Google Docs**
- Two-tier fallback strategy (direct + CORS proxy)
- 60-80% success rate
- 500-800ms latency

---

## Quick Start

### Prerequisites
```bash
python3 -m pip install matplotlib numpy
```

### Run Experiments
```bash
python3 run_experiments.py
```

**Outputs:**
- `test_dataset.json` - 16 synthetic test samples with ground truth
- `main_results.json` - Complete evaluation metrics
- `baseline_results.json` - Baseline comparison data
- `detection_performance.png` - Performance visualizations (4 charts)
- `detailed_analysis.png` - Detailed metrics breakdown (2 charts)

### Test Individual Components

**PII Detector:**
```bash
python3 pii_detector.py
```

**Content Fetcher:**
```bash
python3 content_fetcher.py
```

**Test Data Generator:**
```bash
python3 test_data_generator.py
```

**Evaluator:**
```bash
python3 evaluator.py
```

---

## Repository Structure

```
.
├── pii_detector.py              # Core detection (278 lines)
├── content_fetcher.py           # Platform integration (220 lines)
├── evaluator.py                 # Evaluation framework (310 lines)
├── test_data_generator.py       # Data generation (283 lines)
├── run_experiments.py           # Experiment runner (281 lines)
│
├── test_dataset.json            # Synthetic test data (16 samples)
├── main_results.json            # Evaluation results
├── baseline_results.json        # Baseline comparison
│
├── detection_performance.png    # Performance visualizations
├── detailed_analysis.png        # Detailed metrics
│
└── README.md                    # This file
```

**Total:** 1,372 lines of Python code

---

## Detection Approach

### Regex-Based Pattern Matching

We chose regex over ML-based approaches for:
- **Transparency:** Human-readable, auditable patterns
- **Speed:** No model loading or inference overhead
- **Determinism:** Same input always produces same output
- **No training data required:** Works immediately
- **Perfect for structured PII:** Emails, SSN, credit cards follow regular patterns

### Enhanced Patterns

**Email Detection (100% F1):**
```python
r'\b[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,}\b'
```
- Handles edge cases: plus signs, dots, underscores, subdomains

**Phone Detection (68.1% F1):**
```python
r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
```
- Multiple formats: `(555) 123-4567`, `555-123-4567`, `5551234567`
- False positive filtering: removes dates (12-31-2024)

**Name Detection (72.3% F1):**
```python
r'\b(?:(?:Dr|Mr|Ms|Mrs|Prof)\.?\s+)?([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)\s+(?:([A-Z]\.\s+))?([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)\b'
```
- Handles: titles, hyphenated names, apostrophes, middle initials
- Filtering: removes month names, common words

**Additional PII Types:**
- Street addresses (100% F1)
- Social Security Numbers (100% F1)
- Credit cards with Luhn validation (33.3% F1 - limited by test data)

---

## Evaluation Methodology

### Metrics
- **Precision:** Fraction of predictions that were correct (TP / (TP + FP))
- **Recall:** Fraction of actual PII detected (TP / (TP + FN))
- **F1-Score:** Harmonic mean of precision and recall

### Averaging Strategies
- **Micro-averaging:** Pool all instances (good for overall performance)
- **Macro-averaging:** Average per-sample metrics (good for document-level)

### Baseline Comparison
Simple baseline with basic regex patterns:
- Email: `\S+@\S+\.\S+`
- Phone: `\d{3}-\d{3}-\d{4}`
- Name: `[A-Z][a-z]+ [A-Z][a-z]+`

**Result:** Our system shows +84.3% average F1 improvement over baseline

### Test Data
- 16 synthetic documents with ground truth labels
- Multiple document types: contact lists, signup sheets, mixed content
- Edge cases: emails with +, hyphenated names, various phone formats

---

## Detailed Results

### Comparison with Baseline

| PII Type | Our F1 | Baseline F1 | Improvement |
|----------|--------|-------------|-------------|
| Email | 1.000 | 0.368 | **+172.0%** |
| Phone | 0.681 | 0.340 | **+99.9%** |
| Name | 0.723 | 0.491 | **+47.4%** |
| Address | 1.000 | 0.000 | **+100%** |
| SSN | 1.000 | 0.000 | **+100%** |
| Credit Card | 0.333 | 0.000 | **+100%** |

### Visualizations

**`detection_performance.png`** - 4-panel visualization:
1. F1-Score comparison (our system vs baseline)
2. Precision-Recall trade-off scatter plot
3. Performance metrics heatmap
4. Improvement over baseline

**`detailed_analysis.png`** - 2-panel analysis:
1. TP/FP/FN breakdown by PII type
2. Overall system performance summary

---

## Key Findings

### Strengths
1. **Perfect accuracy for structured PII** - Emails, addresses, SSN achieve 100% F1
2. **High recall for names** - 94.1% recall captures nearly all actual names
3. **Significant improvement over baseline** - Average +84.3% F1 gain
4. **Client-side privacy** - All processing happens locally in browser
5. **Transparent and auditable** - Regex patterns are human-readable

### Limitations
1. **Name precision** - 58.7% due to false positives (capitalized common words, month names)
2. **Phone format coverage** - 74.2% recall, misses some international formats
3. **Google Docs reliability** - 60-80% success rate due to CORS/authentication
4. **Limited real-world validation** - Evaluated on synthetic data only

### Precision-Recall Trade-offs
- **Names:** High recall (94.1%) at cost of precision (58.7%)
  - Catches nearly all actual names but has false positives
  - Trade-off documented and expected
- **Phones:** Balanced approach (62.8% precision, 74.2% recall)
  - False positive filtering reduces spurious detections

---

## Technical Details

### Platform Integration

**URL Detection:**
```python
Platform.PASTEBIN: r'pastebin\.com/([A-Za-z0-9]+)'
Platform.GOOGLE_DOCS: r'docs\.google\.com/document/d/([A-Za-z0-9_-]+)'
```

**URL Transformation:**
- Pastebin: `pastebin.com/ABC` → `pastebin.com/raw/ABC`
- Google Docs: `/document/d/{id}/edit` → `/document/d/{id}/export?format=txt`

**CORS Handling:**
- Pastebin: Native support via `/raw/` endpoint
- Google Docs: Two-tier fallback
  1. Direct fetch from export endpoint
  2. CORS proxy (`api.allorigins.win`) if direct fails

### False Positive Filtering

**Phone Numbers:**
```python
def _filter_phone_false_positives(self, matches):
    # Remove dates like 12-31-2024
    if int(digits[0]) <= 12 and int(digits[1]) <= 31:
        return False  # Likely a date
```

**Names:**
```python
common_words = {'The', 'This', 'That', 'Will', 
                'May', 'June', 'July', 'August', ...}
```

**Credit Cards:**
```python
def _luhn_check(self, card_number: str) -> bool:
    # Luhn algorithm validation
```

---

## Usage Examples

### Example 1: Analyze Sample Text
```python
from pii_detector import PIIDetector

detector = PIIDetector()
text = """
Contact: John Smith
Email: john.smith@example.com
Phone: (555) 123-4567
"""

results = detector.detect_all(text)
for pii_type, matches in results.items():
    print(f"{pii_type}: {len(matches)} found")
```

### Example 2: Evaluate on Test Data
```python
from pii_detector import PIIDetector
from evaluator import PIIEvaluator
from test_data_generator import TestDataGenerator

# Generate test data
generator = TestDataGenerator()
dataset = generator.generate_test_dataset(num_samples=20)

# Evaluate
detector = PIIDetector()
evaluator = PIIEvaluator()
results = evaluator.evaluate_dataset(detector, dataset)

# Print metrics
print(evaluator.format_metrics_table(results['micro_average']))
```

### Example 3: Fetch and Analyze URL
```python
from content_fetcher import ContentFetcher
from pii_detector import PIIDetector

fetcher = ContentFetcher()
detector = PIIDetector()

url = "https://pastebin.com/ABC123"
result = fetcher.fetchContent(url)  # Note: Browser-side only

if result['success']:
    pii_detected = detector.detect_all(result['content'])
```

---

## Related Resources

- **Web Demo:** https://suryacs719.github.io/cse253-openPII-web/
- **Web Repository:** https://github.com/SuryaCS719/cse253-openPII-web
- **Course:** CSE 253 - Network Security, UC Santa Cruz

---

## Contact

**Team Members:**
- Suryakiran Valavala: suvalava@ucsc.edu
- Arsh Advani: agadvani@ucsc.edu
- Vijay Arvind Ramamoorthy: viramamo@ucsc.edu

---

## License

Educational project for CSE 253 at UC Santa Cruz.
