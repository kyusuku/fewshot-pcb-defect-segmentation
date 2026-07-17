# Citation and Novelty Inventory

Literature check date: 2026-07-17. Prefer the publication page where one exists;
retain the arXiv link as an openly accessible fallback.

## Required method and benchmark citations

| Item | Primary source | Use in this project |
| --- | --- | --- |
| DINOv2 | [Oquab et al., 2023](https://arxiv.org/abs/2304.07193) and [official repository](https://github.com/facebookresearch/dinov2) | Frozen visual features. Do not claim the backbone or frozen-feature extraction as novel. |
| PatchCore | [Roth et al., 2021](https://arxiv.org/abs/2106.08265) | Conceptual source for the Wide-ResNet memory-bank/coreset reference. Call the repository implementation **PatchCore-style**, not an exact reproduction. |
| AnomalyDINO | [Damm et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html) | Establishes training-free one/few-shot DINOv2 nearest-neighbor anomaly detection. Do not claim this scoring paradigm as novel. |
| SAM 2 | [Ravi et al., 2024](https://arxiv.org/abs/2408.00714) and [official repository](https://github.com/facebookresearch/sam2) | Frozen promptable mask generation. Do not claim promptable segmentation as novel. |
| VisA | [Zou et al., ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2149_ECCV_2022_paper.php) and [official repository](https://github.com/amazon-science/spot-diff) | Main segmentation benchmark; only `pcb1` through `pcb4` are in the frozen study. |
| DeepPCB | [Tang et al., 2019](https://arxiv.org/abs/1902.06197) and [official repository](https://github.com/tangsanli5201/DeepPCB) | Secondary localization dataset. Its box annotations are not segmentation ground truth and are not part of the main frozen matrix. |

## Closest positioning references

| Work | Primary source | Boundary relative to this project |
| --- | --- | --- |
| Segment Any Anomaly+ | [Cao et al., 2023](https://arxiv.org/abs/2305.10724) | Training-free zero-shot anomaly segmentation with SAM-era foundation-model prompting and multimodal/domain priors; not few-shot normal-only DINOv2-to-SAM2 PCB evaluation. |
| SAM-LAD | [Peng et al., 2024](https://arxiv.org/abs/2406.00625) | Uses SAM object masks and reference features for zero-shot logical anomaly detection; different anomaly type and matching formulation. |
| GFDS | [Liu et al., 2025](https://arxiv.org/abs/2502.01216) | Few-shot defect segmentation study spanning feature matching and SAM2 video tracking; uses few-shot semantic defect support rather than this normal-only anomaly protocol. |
| SubspaceAD | [Lendering et al., CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html) | Recent training-free few-shot DINOv2 anomaly detection using PCA subspace residuals; no SAM2 refinement bridge. |
| Turbulence-robust multi-signal segmentation | [Peng et al., 2026](https://arxiv.org/abs/2605.29292) | A training-free DINOv2/background-anomaly/SAM2 refinement pipeline for turbulent video, not industrial few-shot anomaly segmentation. It confirms that merely combining these foundation models is not itself the claim. |

## Scoped contribution statement

The defensible contribution is a controlled PCB-specific study of a training-free
bridge from multi-scale, few-shot normal-only DINOv2 anomaly proposals to guided
SAM2 masks, followed by an exact anomaly-consistent intersection. The evidence
supports the intersection method relative to the calibrated multi-scale anomaly
proposal; it does not establish that multi-scale DINOv2 alone is better than
single-scale DINOv2, and it does not support a state-of-the-art claim.

Avoid “first,” “novel backbone,” “new anomaly scoring,” “new SAM2,” or broad
state-of-the-art language. Describe the contribution as the protocol, proposal-to-
prompt bridge, exact anomaly-consistency constraint, and repeated-measures PCB
evidence.

## Frozen implementation attribution

- Frozen experiment source commit: `ff4c07208278376b27a4954d560c715f51453c5e`.
- SAM2 source revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- SAM2.1 Hiera Tiny checkpoint SHA-256:
  `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
- SAM2 model-config SHA-256:
  `f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d`.
- DINOv2 ViT-S/14 model-state SHA-256:
  `45d45b4cf4582a5664586cd1d36f1d0d0c8bdeb9152647227765990e950b46df`.
- The original Torch Hub cache did not retain Git metadata. The documented
  official source-equivalent revision is
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; describe it as source-equivalent,
  not as directly retained original provenance.
- Torchvision runtime: `0.20.1+cu121`; the PatchCore-style extractor uses the
  repository's documented frozen Wide-ResNet implementation.

All runtime and model identities above must remain traceable to
`docs/evidence/generated/runtime_provenance.json` after finalization.
