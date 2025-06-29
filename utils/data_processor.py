import json
import re
from typing import Dict, List, Any


class VATIKADataProcessor:
    def __init__(self):
        self.domains = [
            'ganga_aarti', 'cruise', 'food_court', 'public_toilet',
            'kund', 'museum', 'general', 'ashram', 'temple', 'travel'
        ]

    def load_json_data(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}

    def extract_contexts_and_qas(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract contexts and QAs from the dataset"""
        extracted_data = []

        if 'domains' not in data:
            return extracted_data

        for domain_data in data['domains']:
            domain = domain_data['domain']

            for context_data in domain_data['contexts']:
                context = context_data['context']
                qas = context_data['qas']

                extracted_data.append({
                    'domain': domain,
                    'context': context,
                    'qas': qas
                })

        return extracted_data

    def preprocess_text(self, text: str) -> str:
        """Preprocess Hindi text"""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Hindi characters
        text = re.sub(r'[^\w\s\u0900-\u097F।]', ' ', text)
        return text.strip()

    def create_training_examples(self, contexts_qas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create training examples for the model"""
        training_examples = []

        for item in contexts_qas:
            domain = item['domain']
            context = self.preprocess_text(item['context'])

            for qa in item['qas']:
                question = self.preprocess_text(qa['question'])
                answer = self.preprocess_text(qa['answer'])

                training_examples.append({
                    'id': qa['id'],
                    'domain': domain,
                    'context': context,
                    'question': question,
                    'answer': answer
                })

        return training_examples