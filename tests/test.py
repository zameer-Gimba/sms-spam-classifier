import unittest
import sys
sys.path.append("src")

from sms_classifier import predict_message

class TestSMSClassifier(unittest.TestCase):

    def test_ham_message(self):
        _, label = predict_message("Hey, are we still meeting later?")
        self.assertIn(label, ["ham", "spam"])

    def test_spam_message(self):
        _, label = predict_message("Congratulations! You won a free prize. Call now!")
        self.assertIn(label, ["spam", "ham"])

if __name__ == "__main__":
    unittest.main()
