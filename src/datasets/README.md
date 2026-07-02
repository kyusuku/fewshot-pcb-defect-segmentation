# Datasets

Dataset loaders for the two project benchmarks:

- `visa.py`: VisA PCB subsets with image labels and binary segmentation masks.
- `deeppcb.py`: DeepPCB template/test pairs with bounding boxes.
- `sampling.py`: deterministic k-shot normal support sampling.
- `types.py`: shared record and box dataclasses.

DeepPCB boxes are not true segmentation masks. Use pseudo-masks only for prompt construction or qualitative visualization.
