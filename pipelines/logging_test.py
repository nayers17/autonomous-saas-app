# pipelines/logging_test.py

import logging

# Configure logging
logging.basicConfig(filename="../data/logs/test.log", level=logging.INFO)


def dummy_function():
    logging.info("Dummy function called.")
    return "Test completed"


if __name__ == "__main__":
    dummy_function()
