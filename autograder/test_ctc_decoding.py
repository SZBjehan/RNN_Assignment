import numpy as np
import sys, os
import pickle

from test import Test

sys.path.append("CTC")

from CTCDecoding import GreedySearchDecoder

# DO NOT CHANGE -->
isTesting = True
EPS = 1e-20
# -->


class SearchTest(Test):
    def __init__(self):
        pass
        # SEED = 2023
        # np.random.seed(SEED)

    def test_greedy_search(self):
        SEED = 11785
        np.random.seed(11785)
        y_rands = np.random.uniform(EPS, 1.0, (4, 10, 1))
        y_sum = np.sum(y_rands, axis=0)
        y_probs = y_rands / y_sum
        SymbolSets = ["a", "b", "c"]

        expected_results = np.load(
            os.path.join("autograder",  "data", "greedy_search.npy"),
            allow_pickle=True,
        )
        ref_best_path, ref_score = expected_results

        decoder = GreedySearchDecoder(SymbolSets)
        user_best_path, user_score = decoder.decode(y_probs)

        if isTesting:
            try:
                assert user_best_path == ref_best_path
            except Exception as e:
                print("Best path does not match")
                print("Your best path:   ", user_best_path)
                print("Expected best path:", ref_best_path)
                return False

            try:
                assert user_score == float(ref_score)
            except Exception as e:
                print("Best Score does not match")
                print("Your score:    ", user_score)
                print("Expected score:", ref_score)
                return False

        # Use to save test data for next semester
        if not isTesting:
            results = [user_best_path, user_score]
            np.save(os.path.join('autograder', 
                             'data', 'greedy_search.npy'), results, allow_pickle=True)

        return True

    def run_test(self):
        # Test Greedy Search
        self.print_name("Section 5.1 - Greedy Search")
        greedy_outcome = self.test_greedy_search()
        self.print_outcome("Greedy Search", greedy_outcome)
        if greedy_outcome == False:
            self.print_failure("Greedy Search")
            return False

        return True
