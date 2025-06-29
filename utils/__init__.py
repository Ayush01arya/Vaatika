"""
VATIKA Chatbot Utilities Package

This package contains utility modules for the VATIKA tourism chatbot:
- data_processor: Handles data loading and preprocessing
- retriever: Implements advanced context retrieval strategies

Author: VATIKA Development Team
Version: 1.0.0
"""

from .data_processor import VATIKADataProcessor
from .retriever import VATIKARetriever, AdvancedVATIKARetriever

__version__ = "1.0.0"
__author__ = "VATIKA Development Team"

# Export main classes
__all__ = [
    'VATIKADataProcessor',
    'VATIKARetriever',
    'AdvancedVATIKARetriever'
]

# Package-level configuration
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
SUPPORTED_DOMAINS = [
    'ganga_aarti',
    'cruise',
    'food_court',
    'public_toilet',
    'kund',
    'museum',
    'general',
    'ashram',
    'temple',
    'travel'
]

# Hindi language configuration
HINDI_STOPWORDS = [
    'का', 'के', 'की', 'में', 'से', 'को', 'और', 'है', 'हैं', 'था', 'थे', 'थी',
    'होना', 'होने', 'वाला', 'वाले', 'वाली', 'जो', 'जिसे', 'जिसका', 'जिसके',
    'यह', 'वह', 'ये', 'वे', 'इस', 'उस', 'इन', 'उन', 'पर', 'तक', 'लिए',
    'साथ', 'बाद', 'पहले', 'द्वारा', 'रूप', 'तरह', 'प्रकार', 'अपने', 'अपना'
]

# Common Hindi question patterns
QUESTION_PATTERNS = {
    'what': ['क्या', 'कौन', 'कौन सा', 'कौन से'],
    'where': ['कहां', 'कहाँ', 'किस जगह', 'किस स्थान'],
    'when': ['कब', 'किस समय', 'कितने बजे'],
    'how': ['कैसे', 'किस तरह', 'किस प्रकार'],
    'why': ['क्यों', 'किसलिए', 'किस कारण'],
    'which': ['कौन सा', 'कौन से', 'किस'],
    'how_much': ['कितना', 'कितनी', 'कितने'],
    'how_many': ['कितने', 'कितनी संख्या में']
}

# Tourism-specific keywords
TOURISM_KEYWORDS = {
    'religious': ['मंदिर', 'आरती', 'पूजा', 'दर्शन', 'प्रसाद', 'तीर्थ'],
    'places': ['घाट', 'कुंड', 'संग्रहालय', 'आश्रम', 'स्थान'],
    'travel': ['जाना', 'पहुंचना', 'रास्ता', 'दूरी', 'समय', 'किराया'],
    'facilities': ['होटल', 'भोजन', 'शौचालय', 'पार्किंग', 'गाइड'],
    'activities': ['क्रूज़', 'नाव', 'घूमना', 'देखना', 'फोटो']
}


def get_package_info():
    """Get package information"""
    return {
        'name': 'VATIKA Utils',
        'version': __version__,
        'author': __author__,
        'supported_domains': SUPPORTED_DOMAINS,
        'default_model': DEFAULT_EMBEDDING_MODEL
    }


def validate_domain(domain: str) -> bool:
    """Validate if domain is supported"""
    return domain.lower() in [d.lower() for d in SUPPORTED_DOMAINS]


def preprocess_hindi_text(text: str) -> str:
    """Basic Hindi text preprocessing"""
    import re

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)

    # Remove English punctuation but keep Hindi punctuation
    text = re.sub(r'[^\w\s\u0900-\u097F।॥]', ' ', text)

    # Convert to lowercase (for English parts)
    # Hindi doesn't have case distinction

    return text.strip()


def extract_question_type(question: str) -> str:
    """Extract question type from Hindi question"""
    question_lower = question.lower()

    for q_type, patterns in QUESTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in question_lower:
                return q_type

    return 'general'


def get_domain_keywords(domain: str) -> list:
    """Get relevant keywords for a domain"""
    domain_keyword_mapping = {
        'temple': ['मंदिर', 'देवालय', 'पूजा', 'दर्शन', 'प्रसाद'],
        'ghat': ['घाट', 'तट', 'गंगा', 'स्नान'],
        'aarti': ['आरती', 'पूजा', 'गंगा', 'शाम'],
        'food': ['भोजन', 'खाना', 'रेस्टोरेंट', 'स्वादिष्ट'],
        'travel': ['यात्रा', 'जाना', 'पहुंचना', 'रास्ता', 'दूरी'],
        'museum': ['संग्रहालय', 'कलाकृति', 'इतिहास'],
        'ashram': ['आश्रम', 'साधना', 'योग', 'ध्यान'],
        'kund': ['कुंड', 'तालाब', 'जल', 'स्नान'],
        'cruise': ['क्रूज़', 'नाव', 'गंगा', 'सवारी'],
        'toilet': ['शौचालय', 'टॉयलेट', 'सुविधा']
    }

    return domain_keyword_mapping.get(domain.lower(), [])


# Logging configuration
import logging


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vatika_chatbot.log', encoding='utf-8')
        ]
    )

    # Set specific log levels for different modules
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)


# Performance monitoring
import time
from functools import wraps


def performance_monitor(func):
    """Decorator to monitor function performance"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        logger = logging.getLogger(__name__)
        logger.info(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")

        return result

    return wrapper


# Error handling utilities
class VATIKAError(Exception):
    """Base exception class for VATIKA chatbot"""
    pass


class DataLoadError(VATIKAError):
    """Raised when data loading fails"""
    pass


class ModelLoadError(VATIKAError):
    """Raised when model loading fails"""
    pass


class RetrievalError(VATIKAError):
    """Raised when retrieval fails"""
    pass


# Configuration management
class Config:
    """Configuration management class"""

    def __init__(self):
        self.embedding_model = DEFAULT_EMBEDDING_MODEL
        self.max_context_length = 512
        self.retrieval_top_k = 5
        self.similarity_threshold = 0.3
        self.cache_size = 1000
        self.batch_size = 32

    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

    def get_config(self):
        """Get current configuration"""
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }


# Default configuration instance
default_config = Config()