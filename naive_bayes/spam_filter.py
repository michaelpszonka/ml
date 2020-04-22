from typing import Set, NamedTuple, Tuple, Dict, Iterable
import re
from collections import defaultdict
import math


def tokenize(text) -> Set[str]:
    text = text.lower() #convert to lower
    all_words = re.findall("[a-z0-9']+", text) #extract words
    return set(all_words) #remove dupels

class Message(NamedTuple):
    text: str
    is_spam: bool

class NaiveBayesClassifier:
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha #smooting factor

        self.tokens: Set[str] = set()
        self.token_spam_count: Dict[str, int] = defaultdict(int)
        self.token_ham_count: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        #count messages
        for message in messages:
            if(message.is_spam):
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            #increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if(message.is_spam):
                    self.token_spam_count[token] += 1
                else:
                    self.token_ham_count[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """returns P(token | spam) and P(token | ham)"""
        spam = self.token_spam_count[token]
        ham = self.token_ham_count[token]

        p_token_span = (spam + self.alpha) / (self.spam_messages + 2 * self.alpha)
        p_token_ham = (ham + self.alpha) / self.ham_messages + 2 * self.alpha

        return p_token_span, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_probability_if_spam = log_probability_if_ham = 0.0

        #iterate through vocabularyh

        for token in self.tokens:
            probability_if_spam, probability_if_ham = self._probabilities(token)

            #if token is in message, add the log probability of seeing it
            if(token in text_tokens):
                log_probability_if_spam += math.log(probability_if_spam)
                log_probability_if_ham += math.log(probability_if_ham)

            #otherwise add the log probability of not seeing it (1- log_probability_of_seeing_it
            else:
                log_probability_if_spam += math.log(1.0 - probability_if_spam)
                log_probability_if_ham += math.log(1.0 - probability_if_ham)

        probability_if_spam = math.exp(log_probability_if_spam)
        probability_if_ham = math.exp(log_probability_if_ham)
        return probability_if_spam / (probability_if_spam + probability_if_ham)

if __name__ == '__main__':
    messages = [Message("spam rules", is_spam=True),
                Message("ham rules", is_spam=False),
                Message("hello ham", is_spam=False)]

    nb = NaiveBayesClassifier()
    nb.train(messages)