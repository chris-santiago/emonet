import sys

import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from emonet import DATA_DIR, ROOT
from emonet.data_loader import TQRegressionDataset
from emonet.model import EmotionRegressor


def get_saved(emotion: str):
    return ROOT.joinpath('saved_models', f'{emotion}.ckpt')


if __name__ == '__main__':
    try:
        emotion = sys.argv[1]  # if running via CLI
    except IndexError:
        emotion = 'sadness'  # if running via IDE

    model = EmotionRegressor.load_from_checkpoint(get_saved(emotion))

    test = TQRegressionDataset(
        DATA_DIR.joinpath('test.json'),
        DATA_DIR,
        emotion=emotion
    )
    dl = DataLoader(test, 1)

    trainer = pl.Trainer(max_epochs=1)
    preds = trainer.predict(model, dataloaders=dl)
    actuals = [batch[1] for _, batch in enumerate(dl)]

    results = pd.DataFrame({
        'actual_score': [x.item() for x in actuals],
        'predicted_score': [x.item() for x in preds]
    })

    results.dropna(inplace=True)  # NaNs come from samples < 8second duration
    results['has_emotion_actual'] = results['actual_score'] >= 1.5
    results['has_emotion_predicted'] = results['predicted_score'] >= 1.5
    mae = mean_absolute_error(results['actual_score'], results['predicted_score'])
    prec = precision_score(results['has_emotion_actual'], results['has_emotion_predicted'])
    rec = recall_score(results['has_emotion_actual'], results['has_emotion_predicted'])
    acc = accuracy_score(results['has_emotion_actual'], results['has_emotion_predicted'])
    cm = confusion_matrix(results['has_emotion_actual'], results['has_emotion_predicted'])

    with open(f'{emotion}_results.txt', 'w') as fp:
        fp.write(results.to_string())
        fp.writelines(['\n']*2)
        fp.write(f'Mean Absolute Error (MAE): {round(mae, 4)} \n')
        fp.write(f'Precision: {round(prec, 4)} \n')
        fp.write(f'Recall: {round(rec, 4)} \n')
        fp.write(f'Accuracy: {round(acc, 4)} \n')
        fp.write(f'Confusion Matrix: \n {str(cm)}')