def run():
    # TODO: Main script for running spec decoding eval
    # Should take command-line args such as:
    # - source (ntrex | totoeba | other)
    # - language code
    # - spec decoding setting (greedy, eagle, etc)
    # - draft model setting (n-gram, distill, etc)
    # - draft model name (if already trained)
    # Then do the following:
    # 1. Load the appropriate evaluation dataset
    # 2. If no draft model is provided, train the draft model
    # 3. Run according to the setting and log metrics to wandb
    pass


if __name__ == "__main__":
    run()
