import numpy as np
import torch
from tqdm import tqdm


def predict_result(model, device, target_cols, test, test_loader, batch_size=32):
    test_preds = np.zeros((len(test), len(target_cols)))

    model.eval()
    tk0 = tqdm(enumerate(test_loader))
    for idx, x_batch in tk0:
        with torch.no_grad():
            outputs = model(
                input_ids=x_batch[0].to(device),
                labels=None,
                attention_mask=x_batch[1].to(device),
                token_type_ids=x_batch[2].to(device),
            )
            predictions = outputs[0]
            test_preds[idx * batch_size : (idx + 1) * batch_size] = (
                predictions.detach().cpu().squeeze().numpy()
            )

    output = torch.sigmoid(torch.tensor(test_preds)).numpy()
    return output


# def main():
#     test_loader = get_test_loader(batch_size=32)
#     model = SimpleClassifier()

#     checkpoint = torch.load("../models/simple_model_0.55.pt", weights_only=True)
#     model.load_state_dict(checkpoint)

#     test_dataset = init_dataset("../data/test_labeled")
#     test_loader = init_dataloader(test_dataset, 128)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     infer_model(model, test_loader, device)


# if __name__ == "__main__":
#     main()
