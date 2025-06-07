from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytorch_lightning as pl
import torch
from transformers import BertTokenizer

from qa_labeling.pl_modules.model import QALabler
from qa_labeling.utils import compute_inpute_simple


class InferenceModel(pl.LightningModule):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = QALabler.load_from_checkpoint(model_path)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
    ):
        out = self.model(
            input_ids,
            attention_mask,
            token_type_ids,
        )
        preds = torch.sigmoid(out.squeeze())
        return preds


def main():
    model = InferenceModel(Path("./models/epoch=02-val_loss=0.4259-v2.ckpt"))
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    dummy_text = "This is dummy text"
    dummy_input = compute_inpute_simple(dummy_text, tokenizer, 290)

    torch.onnx.export(
        model,
        dummy_input,
        Path("./qa_labeling/qa_model.onnx"),
        export_params=True,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "token_type_ids": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def check_onnx(onnx_model_path: str):
    dummy_text = "This is dummy text"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    dummy_input = compute_inpute_simple(dummy_text, tokenizer, 290)
    ort_sess = ort.InferenceSession(onnx_model_path)
    outputs = ort_sess.run(
        None,
        {
            "input_ids": dummy_input[0].numpy().astype(np.int64),
            "attention_mask": dummy_input[1].numpy().astype(np.int64),
            "token_type_ids": dummy_input[2].numpy().astype(np.int64),
        },
    )

    print(outputs)
    print(outputs[0].shape)


if __name__ == "__main__":
    main()
    check_onnx("./qa_labeling/qa_model.onnx")
