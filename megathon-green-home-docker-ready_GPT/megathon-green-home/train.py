import json
from app.ml import train_and_evaluate

if __name__ == '__main__':
    metrics=train_and_evaluate()
    print(json.dumps(metrics,indent=2))
