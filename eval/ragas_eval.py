import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

def run_ragas(df: pd.DataFrame):
    result = evaluate(df, metrics=[faithfulness, answer_relevancy, context_precision])
    print(result)
    return result

if __name__ == "__main__":
    df = pd.read_csv("eval/evalset.csv")
    run_ragas(df)
