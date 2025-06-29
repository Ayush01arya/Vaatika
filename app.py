from flask import Flask, render_template, request, jsonify
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from transformers import pipeline
import torch

app = Flask(__name__)


class ImprovedVATIKAChatbot:
    def __init__(self):
        print("🚀 Initializing VATIKA Chatbot...")

        # Load multilingual embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Embedding model loaded")

        # Load QA model with better error handling
        self.qa_pipeline = self.load_qa_model()

        # Initialize data structures
        self.contexts = []
        self.context_embeddings = None
        self.qa_pairs = []  # Store all Q&A pairs separately
        self.qa_embeddings = None

        # Load and process data
        self.load_data()
        print(f"✅ Loaded {len(self.contexts)} contexts and {len(self.qa_pairs)} Q&A pairs")

    def load_qa_model(self):
        """Load QA model with fallback options"""
        models_to_try = [
            "deepset/xlm-roberta-base-squad2",
            "distilbert-base-multilingual-cased",
            "deepset/minilm-uncased-squad2"
        ]

        for model_name in models_to_try:
            try:
                print(f"🔄 Trying to load QA model: {model_name}")
                qa_pipeline = pipeline(
                    "question-answering",
                    model=model_name,
                    tokenizer=model_name,
                    device=-1  # Use CPU
                )
                print(f"✅ Successfully loaded: {model_name}")
                return qa_pipeline
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue

        print("⚠️ Could not load any QA model, using fallback")
        return None

    def load_data(self):
        """Load and preprocess VATIKA dataset with better error handling"""
        try:
            # Check if data files exist
            train_file = 'data/train.json'
            val_file = 'data/validation.json'

            if not os.path.exists(train_file):
                print("❌ Train file not found, creating sample data...")
                self.create_sample_data()

            # Load training data
            with open(train_file, 'r', encoding='utf-8') as f:
                train_data = json.load(f)

            all_data = train_data.get('domains', [])

            # Try to load validation data
            if os.path.exists(val_file):
                try:
                    with open(val_file, 'r', encoding='utf-8') as f:
                        val_data = json.load(f)
                    all_data.extend(val_data.get('domains', []))
                    print("✅ Validation data loaded")
                except Exception as e:
                    print(f"⚠️ Could not load validation data: {e}")

            # Process data
            self.process_data(all_data)

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.create_fallback_data()

    def process_data(self, domains_data):
        """Process loaded data and create embeddings"""
        all_contexts = []
        all_qas = []

        for domain_data in domains_data:
            domain = domain_data.get('domain', 'unknown')

            for context_data in domain_data.get('contexts', []):
                context_text = context_data.get('context', '')
                qas = context_data.get('qas', [])

                if context_text.strip():  # Only add non-empty contexts
                    context_info = {
                        'domain': domain,
                        'context': context_text,
                        'qas': qas
                    }
                    all_contexts.append(context_info)

                    # Extract Q&A pairs
                    for qa in qas:
                        question = qa.get('question', '').strip()
                        answer = qa.get('answer', '').strip()

                        if question and answer:
                            qa_info = {
                                'question': question,
                                'answer': answer,
                                'domain': domain,
                                'context': context_text
                            }
                            all_qas.append(qa_info)

        self.contexts = all_contexts
        self.qa_pairs = all_qas

        # Create embeddings
        if self.contexts:
            print("🔄 Creating context embeddings...")
            context_texts = [ctx['context'] for ctx in self.contexts]
            self.context_embeddings = self.embedding_model.encode(context_texts, show_progress_bar=True)

        if self.qa_pairs:
            print("🔄 Creating Q&A embeddings...")
            qa_questions = [qa['question'] for qa in self.qa_pairs]
            self.qa_embeddings = self.embedding_model.encode(qa_questions, show_progress_bar=True)

    def create_sample_data(self):
        """Create sample data if original data is not available"""
        sample_data = {
            "domains": [
                {
                    "domain": "varanasi_temples",
                    "contexts": [
                        {
                            "context": "काशी विश्वनाथ मंदिर वाराणसी का सबसे प्रसिद्ध और पवित्र मंदिर है। यह भगवान शिव को समर्पित है और गंगा नदी के पश्चिमी तट पर स्थित है। यह 12 ज्योतिर्लिंगों में से एक है और हिंदू धर्म में अत्यंत महत्वपूर्ण माना जाता है।",
                            "qas": [
                                {
                                    "question": "काशी विश्वनाथ मंदिर कहाँ स्थित है?",
                                    "answer": "काशी विश्वनाथ मंदिर वाराणसी में गंगा नदी के पश्चिमी तट पर स्थित है।"
                                },
                                {
                                    "question": "काशी विश्वनाथ मंदिर किसे समर्पित है?",
                                    "answer": "काशी विश्वनाथ मंदिर भगवान शिव को समर्पित है।"
                                },
                                {
                                    "question": "क्या काशी विश्वनाथ ज्योतिर्लिंग है?",
                                    "answer": "हाँ, काशी विश्वनाथ मंदिर 12 ज्योतिर्लिंगों में से एक है।"
                                }
                            ]
                        }
                    ]
                },
                {
                    "domain": "varanasi_ghats",
                    "contexts": [
                        {
                            "context": "दशाश्वमेध घाट वाराणसी का सबसे प्रसिद्ध और मुख्य घाट है। यहाँ प्रतिदिन शाम को भव्य गंगा आरती का आयोजन होता है। यह घाट अत्यंत पवित्र माना जाता है और हजारों श्रद्धालु और पर्यटक यहाँ आते हैं।",
                            "qas": [
                                {
                                    "question": "दशाश्वमेध घाट पर आरती कब होती है?",
                                    "answer": "दशाश्वमेध घाट पर प्रतिदिन शाम को गंगा आरती होती है।"
                                },
                                {
                                    "question": "वाराणसी का सबसे प्रसिद्ध घाट कौन सा है?",
                                    "answer": "दशाश्वमेध घाट वाराणसी का सबसे प्रसिद्ध घाट है।"
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        os.makedirs('data', exist_ok=True)
        with open('data/train.json', 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

        print("✅ Sample data created")

    def create_fallback_data(self):
        """Create minimal fallback data"""
        self.contexts = [{
            'domain': 'general',
            'context': 'वाराणसी भारत का एक प्राचीन और पवित्र शहर है।',
            'qas': [{'question': 'वाराणसी क्या है?', 'answer': 'वाराणसी भारत का एक प्राचीन और पवित्र शहर है।'}]
        }]

        self.qa_pairs = [{'question': 'वाराणसी क्या है?', 'answer': 'वाराणसी भारत का एक प्राचीन और पवित्र शहर है।',
                          'domain': 'general'}]

        context_texts = [ctx['context'] for ctx in self.contexts]
        self.context_embeddings = self.embedding_model.encode(context_texts)

        qa_questions = [qa['question'] for qa in self.qa_pairs]
        self.qa_embeddings = self.embedding_model.encode(qa_questions)

    def find_best_qa_match(self, query, threshold=0.6):
        """Find best matching Q&A pair"""
        if not self.qa_pairs or self.qa_embeddings is None:
            return None

        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.qa_embeddings)[0]

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score > threshold:
            return {
                'qa': self.qa_pairs[best_idx],
                'score': best_score
            }

        return None

    def find_relevant_context(self, query, top_k=3, threshold=0.3):
        """Find most relevant contexts"""
        if not self.contexts or self.context_embeddings is None:
            return []

        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.context_embeddings)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        relevant_contexts = []
        for idx in top_indices:
            if similarities[idx] > threshold:
                relevant_contexts.append({
                    'context': self.contexts[idx],
                    'similarity': similarities[idx]
                })

        return relevant_contexts

    def generate_qa_answer(self, question, context):
        """Generate answer using QA model"""
        if not self.qa_pipeline:
            return None

        try:
            # Truncate context if too long
            max_context_length = 500
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."

            result = self.qa_pipeline(question=question, context=context)

            if result['score'] > 0.15:  # Confidence threshold
                return result['answer']

        except Exception as e:
            print(f"QA Pipeline error: {e}")

        return None

    def get_smart_fallback(self, query):
        """Generate smart fallback responses"""
        query_lower = query.lower()

        # Keywords-based responses
        responses = {
            ('मंदिर',
             'temple'): "वाराणसी में काशी विश्वनाथ मंदिर, संकट मोचन हनुमान मंदिर, दुर्गा मंदिर जैसे प्रसिद्ध मंदिर हैं। किसी विशिष्ट मंदिर के बारे में पूछें।",
            ('घाट',
             'ghat'): "वाराणसी में दशाश्वमेध घाट, मणिकर्णिका घाट, अस्सी घाट जैसे प्रसिद्ध घाट हैं। किसी विशिष्ट घाट के बारे में जानना चाहते हैं?",
            ('आरती', 'aarti'): "गंगा आरती दशाश्वमेध घाट पर प्रतिदिन शाम को होती है। यह बहुत ही मनोहर और भव्य होती है।",
            ('गंगा', 'ganga'): "गंगा नदी वाराणसी की जीवनधारा है। यहाँ लोग स्नान करते हैं और आरती देखते हैं।",
            ('यात्रा', 'travel',
             'घूमना'): "वाराणसी में आप मंदिर, घाट, गलियाँ, और सांस्कृतिक स्थल देख सकते हैं। क्या विशिष्ट जानकारी चाहिए?"
        }

        for keywords, response in responses.items():
            if any(keyword in query_lower for keyword in keywords):
                return response

        return "मुझे वाराणसी के बारे में आपका प्रश्न समझ नहीं आया। कृपया मंदिर, घाट, आरती, या यात्रा के बारे में पूछें।"

    def process_query(self, query):
        """Main query processing function"""
        if not query.strip():
            return "कृपया अपना प्रश्न पूछें।"

        print(f"🔍 Processing query: {query}")

        # Step 1: Try to find direct Q&A match
        qa_match = self.find_best_qa_match(query)
        if qa_match:
            print(f"✅ Found Q&A match with score: {qa_match['score']:.3f}")
            return qa_match['qa']['answer']

        # Step 2: Find relevant contexts
        relevant_contexts = self.find_relevant_context(query)

        if relevant_contexts:
            print(f"✅ Found {len(relevant_contexts)} relevant contexts")

            # Step 3: Try QA model on best context
            best_context = relevant_contexts[0]['context']
            qa_answer = self.generate_qa_answer(query, best_context['context'])

            if qa_answer:
                return qa_answer

            # Step 4: Check for direct Q&As in the context
            for qa in best_context['qas']:
                if self.is_similar_question(query, qa['question']):
                    return qa['answer']

        # Step 5: Smart fallback
        return self.get_smart_fallback(query)

    def is_similar_question(self, q1, q2, threshold=0.7):
        """Check if two questions are similar"""
        try:
            embeddings = self.embedding_model.encode([q1, q2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return similarity > threshold
        except:
            return False


# Initialize improved chatbot
chatbot = ImprovedVATIKAChatbot()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'कृपया कोई संदेश भेजें'}), 400

        # Process the query
        response = chatbot.process_query(user_message)

        # Add debug info in development
        debug_info = {
            'total_contexts': len(chatbot.contexts),
            'total_qas': len(chatbot.qa_pairs),
            'model_loaded': chatbot.qa_pipeline is not None
        }

        return jsonify({
            'response': response,
            'status': 'success',
            'debug': debug_info if app.debug else None
        })

    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({
            'error': f'कुछ गलती हुई है: {str(e)}',
            'status': 'error'
        }), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'contexts_loaded': len(chatbot.contexts),
        'qas_loaded': len(chatbot.qa_pairs),
        'embeddings_ready': chatbot.context_embeddings is not None,
        'qa_model_loaded': chatbot.qa_pipeline is not None
    })


@app.route('/debug')
def debug():
    """Debug endpoint to check data"""
    return jsonify({
        'contexts': len(chatbot.contexts),
        'qa_pairs': len(chatbot.qa_pairs),
        'sample_context': chatbot.contexts[0] if chatbot.contexts else None,
        'sample_qa': chatbot.qa_pairs[0] if chatbot.qa_pairs else None
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)