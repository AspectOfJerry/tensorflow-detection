def non_maximum_suppression(predictions, iou_threshold):
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    keepers = []

    while predictions:
        keeper = predictions.pop(0)
        keepers.append(keeper)

        predictions = [
            pred for pred in predictions
            if IoU(pred['bbox'], keeper['bbox']) < iou_threshold
        ]

    return keepers
