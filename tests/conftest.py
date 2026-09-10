import os
import sys

os.environ["FRAUD_SKIP_DEMOS"] = "1"  # must be set BEFORE fraud_multi_agent is imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
