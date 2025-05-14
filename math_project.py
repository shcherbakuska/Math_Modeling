import re
import pandas as pd
import spacy
from pymorphy3 import MorphAnalyzer
from typing import Dict, List
from itertools import product
import difflib

nlp = spacy.load("ru_core_news_sm")
morph = MorphAnalyzer()

class GlossaryCreator:
    def __init__(self):
        # Список паттернов для поиска терминов
        self.conditional_patterns = [
            # Если ..., то суффикс называется префикс
            r'(?P<definition>Если\s.+?),\s+(?P<term_suffix>[А-Яа-яЁё\s-]+?)\s+называ(?:ется|ются|ют)\s+(?P<term_prefix>[А-Яа-яЁё\s-]+)(?=[.!?](?!\s?[а-я]))',
            # Суффикс называется префикс, если ...
            r'(?P<term_suffix>[А-Яа-яЁё\s-]+?)\s+называ(?:ется|ются|ют)\s+(?P<term_prefix>[А-Яа-яЁё\s-]+?),?\s+если\s+(?P<definition>.+?)(?=[.!?](?!\s?[а-я]))',
        ]

        self.primary_patterns = [
            # Термин — определение
            r'(?P<term>[А-Яа-яЁё\s-]+?)\s*—\s*(?P<definition>.+?)(?=[.!?](?!\s?[а-я]))',
            # Термин (аббревиатура) — определение
            r'(?P<term>[А-Яа-яЁё\s-]+?)\s*\((?P<abbr>[^)]+?)\)\s*—\s*(?P<definition>.+?)(?=[.!?](?!\s?[а-я]))',
            # Под термином понимается ...
            r'[Пп]од\s+(?P<term>[А-Яа-яЁё\s-]+?)\s+(?:будем\s+)?понима[ею]т(ся)?\s+(?P<definition>.+?)(?=[.!?](?!\s?[а-я]))',
            # Термин — это ...
            r'(?P<term>[А-Яа-яЁё\s-]+?)\s*—\s*(?:это|есть)\s+(?P<definition>.+?)(?=[.!?](?!\s?[а-я]))',
            # Термин называется/является/обозначается ...
            r'(?P<term>[А-Яа-яЁё\s-]+?)\s+(?:называ(?:ется|ются|ют)|является|обозначается)\s+(?P<definition>.+?)(?=[.!?](?!\s?[а-я]))',
            # Определение(,) называется термином
            r'(?P<definition>.+?),?\s+называ(?:ется|ются|ют|ется как)?\s+(?P<term>[А-Яа-яЁё\s-]+)(?=[.!?](?!\s?[а-я]))',
            # Определение — это термин
            r'(?P<definition>.+?)\s+—\s+(?:это|есть)\s+(?P<term>[А-Яа-яЁё\s-]+)(?=[.!?](?!\s?[а-я]))',
        ]



    def extract_terms(self, text: str) -> Dict[str, str]:
        glossary = {}

        for pattern in self.conditional_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                suffix = match.group('term_suffix').strip()
                prefix = match.group('term_prefix').strip()
                term = f"{prefix} {suffix}"
                definition = match.group('definition').strip()
                self.term_assembly(term, definition, glossary)

        if not glossary:
            for pattern in self.primary_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    term = match.group('term').strip()
                    definition = match.group('definition').strip()
                    self.term_assembly(term, definition, glossary)

        return glossary

    def term_assembly(self, term: str, definition: str, glossary: Dict[str, str]):
        # Очистка термина от сторонних слов
        term = re.sub(r'\b(которая|который|которое|это|данное|данный|такая|такой|такое|которые|которых|тех|тому|то)\b', '', term, flags=re.IGNORECASE)
        term = re.sub(r'\s{2,}', ' ', term).strip()

        lemmatized_term = self.term_lemmatization(term)
        if lemmatized_term and definition:
            glossary[lemmatized_term] = definition

    def term_lemmatization(self, term: str) -> str:
        doc = nlp(term)
        tokens = list(doc)

        # Нахождение первого существительного
        head_noun = None
        for token in tokens:
            if token.pos_ == "NOUN":
                head_noun = token
                break

        if not head_noun:
            return ' '.join([token.lemma_ for token in tokens])

        # Приведение к нормальной форме
        noun_parse = morph.parse(head_noun.text)[0]
        noun_lemma = noun_parse.normal_form
        noun_gram = {noun_parse.tag.gender, noun_parse.tag.number, 'nomn'}

        result = []

        for token in tokens:
            if token == head_noun:
                result.append(noun_lemma)
            elif token.pos_ == "ADJ" and token.i < head_noun.i:
                adj_parse = morph.parse(token.text)[0]
                dependency = adj_parse.inflect(noun_gram)
                result.append(dependency.word if dependency else adj_parse.normal_form)
            else:
                result.append(token.text)

        return ' '.join(result)

class TermSearcher:
    def __init__(self):
        self.extractor = GlossaryCreator()

    def find_term_variations(self, term: str, text: str) -> List[str]:
        variations = set()

        # Обработка однословных терминов
        tokens = term.split()
        if len(tokens) == 1:
            word = tokens[0]
            parsed = morph.parse(word)[0]
            variations.update({word, parsed.normal_form})
            for form in parsed.lexeme:
                variations.add(form.word)
        else:
            # Обработка многословных терминов
            base_forms = [morph.parse(t)[0].normal_form for t in tokens]
            variations.add(' '.join(base_forms))
            variations.add(term)

            # Генерация различных комбинаций форм слов
            word_forms = []
            for token in tokens:
                parsed = morph.parse(token)[0]
                word_forms.append([f.word for f in parsed.lexeme])

            # Ограничение количества комбинаций
            from itertools import product
            for combo in product(*word_forms[:3]):
                variations.add(' '.join(combo))

        # Поиск в тексте
        found = []
        for variant in variations:
            if variant.lower() in text.lower():
                start_idx = text.lower().find(variant.lower())
                if start_idx != -1:
                    end_idx = start_idx + len(variant)
                    context = text[max(0, start_idx-40):min(len(text), end_idx+40)]
                    found.append(context.strip())

        return found

def load_texts(filename: str) -> List[str]:
    try:
        df = pd.read_csv(filename, header=None, names=['text'])
        return df['text'].dropna().tolist()
    except Exception as e:
        print(f"Ошибка загрузки файла {filename}: {e}")
        return []

def duplicates_elimination(quotes: List[str]) -> List[str]:
    # Удаление дублирующихся цитат на основе текстового сходства
    unique_quotes = []
    for quote in quotes:
        if not any(difflib.SequenceMatcher(None, quote, existing).ratio() >  0.85 for existing in unique_quotes):
            unique_quotes.append(quote)
    return unique_quotes

def creator_evaluation(glossary: Dict[str, str], test_csv: str) -> Dict[str, float]:
    test_terms = set()
    test_texts = load_texts(test_csv)

    for text in test_texts:
        if ':' in text:
            term = text.split(':')[0].strip().lower()
            test_terms.add(term)

    extracted_terms = set(term.lower() for term in glossary.keys())

    true_positives = extracted_terms & test_terms
    false_positives = extracted_terms - test_terms
    false_negatives = test_terms - extracted_terms

    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': len(true_positives),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives)
    }


def searcher_evaluation(mentions: Dict[str, List[str]], standard: int) -> Dict[str, float]:
    if not mentions:
        return {
            'accuracy': 0.0,
            'mentions_quantity': 0,
            'standard': standard
        }

    mentions_quantity = sum(len(quotes) for quotes in mentions.values())
    accuracy = mentions_quantity / standard

    return {
        'accuracy': accuracy,
        'mentions_quantity': mentions_quantity,
        'standard': standard
    }

def main():
    standard = int(input("Введите эталонное количество упоминаний терминов в корпусе: "))
    terms_texts = load_texts('terms.csv')
    corpus_texts = load_texts('corpus.csv')

    # Извлечение терминов и вывод словаря
    extractor = GlossaryCreator()
    glossary = {}
    for text in terms_texts:
        glossary.update(extractor.extract_terms(text))

    print("\nСловарь терминов (термин: определение):")
    print("{")
    for term, definition in glossary.items():
        print(f"    '{term}': '{definition}',")
    print("}")

    # Поиск и вывод упоминаний терминов
    searcher = TermSearcher()
    mentions = {}
    for term in glossary:
        term_mentions = []
        for text in corpus_texts:
            found = searcher.find_term_variations(term, text)
            term_mentions.extend(found)
        if term_mentions:
            mentions[term] = duplicates_elimination(term_mentions)

    print("\nУпоминания терминов в корпусе текста (термин: [цитаты]):")
    print("{")
    for term, quotes in mentions.items():
        print(f"    '{term}': [")
        for quote in quotes:
            print(f"        '{quote}',")
        print("    ],")
    print("}")

    # Оценка эффективности алгоритмов
    if glossary and 'terms_test.csv':
        creator_metrics = creator_evaluation(glossary, 'terms_test.csv')
        print("\nОценка извлечения терминов:")
        print(f"Precision: {creator_metrics['precision']:.2f}")
        print(f"Recall: {creator_metrics['recall']:.2f}")
        print(f"F1-score: {creator_metrics['f1_score']:.2f}")
        print(f"True Positives: {creator_metrics['true_positives']}")
        print(f"False Positives: {creator_metrics['false_positives']}")
        print(f"False Negatives: {creator_metrics['false_negatives']}")

    if mentions:
        searcher_metrics = searcher_evaluation(mentions, standard)
        print("\nОценка поиска упоминаний:")
        print(f"Accuracy: {searcher_metrics['accuracy']:.2f}")
        print(f"Всего найдено упоминаний: {searcher_metrics['mentions_quantity']}/{searcher_metrics['standard']}")



if __name__ == "__main__":
    main()
