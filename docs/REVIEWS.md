# MICCAI 2026 Peer-Review Analysis & Reviewer Concerns (`REVIEWS.md`)

This document provides the complete, structured deconstruction of all reviewer concerns from the **MICCAI 2026 (Submission #2)** peer-review process for our paper: **"360° Novel View Synthesis from Single Chest X-Ray Images via a World Foundation Model"**.

---

## 🗺️ Reviewer Concerns & Technical Solution Matrix

| Concern ID | Reviewer | Core Concern Summary | Target Section in `PROPOSAL.md` | Execution Phase in `EXECUTION.md` |
|:---:|:---:|:---|:---|:---|
| **R2-1** | Reviewer #2 | Generalization to diverse anatomies & pathological variations | Section 1, Section 6 | Phase 3 (VinDr-CXR & MURA Probes) |
| **R2-2** | Reviewer #2 | Comparison with SOTA baselines (MedNeRF, Dx2CT) on OOD split | Section 2 | Phase 1 (TCIA+MELA $\to$ NSCLC Retraining) |
| **R3-1** | Reviewer #3 | Downstream clinical tasks & 3D CT reconstruction value | Section 3 | Phase 1 & Phase 4 |
| **R3-2** | Reviewer #3 | Performance limits, sequential rollouts & hallucination pitfalls | Section 4 | Phase 4 |
| **R3-3** | Reviewer #3 | DRR vs. CXR physical domain gaps & acquisition disparities | Section 5 | Phase 2 (DiffDRR Siddon-Jacob) |
| **R3-4** | Reviewer #3 | Literature positioning against CXR World Models (CheXworld, X-WIN) | Section 7 | Phase 4 |
| **R4-1** | Reviewer #4 | Incremental contribution & medical-specific technical insights | Section 2, Section 7 | Phase 2 & Phase 4 |
| **R4-2** | Reviewer #4 | Motivation for NVS as intermediate 2D-to-3D regularizer | Section 3 | Phase 1 & Phase 4 |

---

## 1. Reviewer #2 Detailed Deconstruction

### 1.1 Concern R2-1: Diverse Anatomies & Pathological Variations
* **Verbatim Reviewer Quote:**
  > *"The evaluation focuses primarily on chest X-rays. Given the claim of transferable geometric representations, experiments on diverse anatomies or pathological variations would strengthen the argument."*
* **Deconstructed Requirements:**
  1. Evaluate model capacity and generalization limits on diverse anatomies outside the chest region.
  2. Provide quantitative or rigorous qualitative validation demonstrating structural and pathological consistency.
* **Vietnamese Analysis:**
  * Phần đánh giá hiện tại chủ yếu trên X-quang ngực, chưa đủ chứng minh tính tổng quát cho tuyên bố "transferable geometric representations".
  * Cần bổ sung đánh giá trên các cơ quan khác (diverse anatomies) hoặc các trường hợp bệnh lý (pathological variations như viêm phổi, tràn dịch màng phổi, tim to).
* **Technical Action Plan:**
  * **Pathology Audit (VinDr-CXR):** Qualitatively audit 3D view-consistency on Cardiomegaly (CTR > 0.5), Pleural Effusion (fluid meniscus rotation), and Pneumonia (depth-plane translation).
  * **Out-of-Domain Probe (Stanford MURA):** Run zero-shot evaluation on hand/wrist musculoskeletal X-rays to establish physical video transfer boundaries (trabecular detail loss).
  * **Anatomical Extension Roadmap:** Formalize adaptation strategies for Musculoskeletal (MURA/MORE), Abdominal (AMOS/AbdomenCT-1K), and Dental (ToothFairy 1 & 2) imaging in `PROPOSAL.md` Section 6.

### 1.2 Concern R2-2: Comparison with SOTA Baselines & OOD Evaluation
* **Verbatim Reviewer Quote:**
  > *"The method is not compared against other relevant state-of-the-art approaches such as MedNeRF and Dx2CT, which are important benchmarks in reconstruction-based and diffusion-based approaches."*
* **Deconstructed Requirements:**
  1. Implement and retrain SOTA baselines on an identical training dataset split.
  2. Conduct a fair, rigorous comparative evaluation using a robust cross-dataset (out-of-distribution) strategy.
* **Vietnamese Analysis:**
  * Thiếu so sánh với MedNeRF (dựa trên NeRF reconstruction) và Dx2CT (dựa trên diffusion).
  * Cần retrain toàn bộ 6 baseline SOTA trên cùng tập dữ liệu OOD.
* **Technical Action Plan:**
  * **Cross-Dataset OOD Protocol:** Train/Val on **TCIA (771 cases) + MELA (525 cases)** = $1,296$ volumes; Test on unseen **NSCLC Radiogenomics (402 cases)**.
  * **Retrain 6 Baselines:** Retrain **Dx2CT**, **SV-DRR**, **MedNeRF**, **NAF**, **PixelNeRF**, and **XRaySyn** under standardized DiffDRR Siddon-Jacob projection rendering.
  * **Quantitative Benchmarking:** Compile PSNR, SSIM, and Inference Latency into Table 1 (`PROPOSAL.md` Section 2.3).

---

## 2. Reviewer #3 Detailed Deconstruction

### 2.1 Concern R3-1: Medical Downstream Tasks & Real-World Impact
* **Verbatim Reviewer Quote:**
  > *"The table 1 evaluates the projection generation quality. But simply generating projections lacks impact on real-world application. The author should clarify what are the downstream tasks that this world foundation models can be applied to and conduct experiments to evaluate performance."*
* **Deconstructed Requirements:**
  1. Clarify concrete medical downstream applications for synthesized 360° projection sequences.
  2. Demonstrate how 360° novel view synthesis acts as an intermediate representation enabling 3D CT volume reconstruction.
* **Vietnamese Analysis:**
  * Việc chỉ tạo ảnh NVS 2D thiếu tác động thực tế trong y tế. Cần chứng minh tính hữu ích bằng downstream tasks cụ thể (như tái tạo khối CT 3D).
* **Technical Action Plan:**
  * **Tomographic Reconstruction Formulations:** Formalize downstream 3D CT reconstruction via Classical Backprojection (FDK) and Beer-Lambert-corrected Neural Radiance Fields (NeRF) in `PROPOSAL.md` Section 3.

### 2.2 Concern R3-2: Performance Limits, Sequential Rollouts & Hallucination Pitfalls
* **Verbatim Reviewer Quote:**
  > *"Generating a sequence of novel projections given a single frontal view is a extremely challenging task. That's probably only a modest PSNR of 23.26 was attained. The authors should discuss solutions to improve performance (such as sequential multi-step rollout methods, incorporating more views as priors, etc.) and potential pitfalls on this topic."*
* **Deconstructed Requirements:**
  1. Discuss architectural pathways to push beyond the current PSNR limits.
  2. Analyze potential pitfalls and risks of generative world models in medical settings (e.g., anatomical hallucinations).
* **Vietnamese Analysis:**
  * Tạo chuỗi góc nhìn từ 1 ảnh đơn là rất khó (PSNR ~23.26dB). Reviewer yêu cầu thảo luận cách cải thiện (multi-step rollout, sparse priors) và rủi ro/cạm bẫy (hallucinations).
* **Technical Action Plan:**
  * **Sequential Causal Rollouts:** Formulate sliding-window autoregressive conditioning $\hat{P}_{\theta_t} = G(P_{\theta_0}, \hat{P}_{\theta_{t-1}}, \dots, \hat{P}_{\theta_{t-k}})$.
  * **Hallucination Clamping:** Detail how the **Frame-Token Replacement Mechanism** ($z_{t_{\text{frontal}}} \leftarrow z_{\text{frontal}}$) acts as a hard boundary condition to clamp generative drift and prevent fake clinical pathology synthesis (`PROPOSAL.md` Section 4).

### 2.3 Concern R3-3: Physical Domain Gaps Between DRRs and CXRs
* **Verbatim Reviewer Quote:**
  > *"There exists huge domain gaps in resolution and appearance between DRRs and CXRs due to differences in acquisition mechanisms. Discussion on how the domain gaps affect final performance is meaningful."*
* **Deconstructed Requirements:**
  1. Analyze physical disparities between simulated Digitally Reconstructed Radiographs (DRRs) and real clinical Chest X-rays (CXRs).
  2. Outline mitigation strategies to bridge this domain gap.
* **Vietnamese Analysis:**
  * Tồn tại domain gap lớn giữa ảnh mô phỏng DRR và ảnh X-quang lâm sàng CXR (nhiễu, phân bố HU, độ phân giải, LUT). Cần phân tích ảnh hưởng đến performance.
* **Technical Action Plan:**
  * **Physics Grounding:** Replace PyTorch3D with GPU-accelerated **DiffDRR Siddon-Jacob raymarching** modeling exact forward attenuation $I = I_0 \exp(-\int \mu \, ds)$.
  * **Domain Adaptation Roadmap:** Propose latent Unsupervised Domain Adaptation (UDA) via Maximum Mean Discrepancy (MMD) to align DRR and MIMIC-CXR feature distributions (`PROPOSAL.md` Section 5).

### 2.4 Concern R3-4: Literature Positioning Against Medical World Models
* **Verbatim Reviewer Quote:**
  > *"This paper lacks discussion on related CXR world models, such as X-WIN CVPR'26 and CheXworld CVPR'25."*
* **Deconstructed Requirements:**
  1. Differentiate Cosmos-Predict2.5 from recent landmark medical world models (`CheXworld` CVPR'25, `X-WIN` CVPR'26).
  2. Clarify why representation-learning models cannot synthesize pixel-level $360^\circ$ novel views.
* **Vietnamese Analysis:**
  * Bài báo chưa thảo luận các CXR world models gần đây như CheXworld (CVPR 2025) và X-WIN (CVPR 2026).
* **Technical Action Plan:**
  * **Comparative Positioning:** Detail that `CheXworld` and `X-WIN` learn discriminative/diagnostic latent embeddings without pixel-level generative decoders, positioning **Cosmos-Predict2.5** as the first explicit pixel-level $360^\circ$ NVS generative world model (`PROPOSAL.md` Section 7).

---

## 3. Reviewer #4 Detailed Deconstruction

### 3.1 Concern R4-1: Technical Contribution & Medical Domain Insights
* **Verbatim Reviewer Quote:**
  > *"The method largely relies on direct fine-tuning of a video foundation model, with limited substantial modifications or domain-specific insights tailored to the medical setting. As a result, the technical contribution appears incremental."*
* **Deconstructed Requirements:**
  1. Highlight medical domain-specific adaptations introduced on top of the base video foundation model.
  2. Prove that adapting natural video priors to medical projection physics requires specialized architectural mechanisms.
* **Vietnamese Analysis:**
  * Phương pháp bị đánh giá là fine-tune đơn thuần từ video foundation model, đóng góp kỹ thuật mang tính incremental.
* **Technical Action Plan:**
  * **Domain-Specific Insights:** Emphasize the **Frame-Token Replacement Mechanism** (anchoring 3D rotational trajectories to frontal CXR latents), spatial-temporal RoPE coordinate transfer from camera dynamics to projection angles, and Beer-Lambert attenuation alignment (`PROPOSAL.md` Section 2 & Section 8).

### 3.2 Concern R4-2: Task Motivation & 3D CT Reconstruction Justification
* **Verbatim Reviewer Quote:**
  > *"The motivation for the task of X-ray novel view synthesis is questionable. In practice, the primary interest lies in recovering the underlying 3D structure (e.g., CT volumes), rather than generating 2D images. While prior work has explained X-ray synthesis as a form of data augmentation for CT reconstruction, this connection is not established in the paper. A stronger justification of the task, along with experiments demonstrating its clinical value (e.g., improvements in CT reconstruction), would be necessary."*
* **Deconstructed Requirements:**
  1. Provide strong clinical justification for 2D X-ray novel view synthesis.
  2. Formulate 360° NVS as a dense regularizing intermediate representation that converts the ill-posed 1-view 3D CT inverse problem into a well-conditioned tomographic reconstruction task.
* **Vietnamese Analysis:**
  * Motivation tạo ảnh 2D NVS chưa đủ thuyết phục vì mục tiêu thực tế là khôi phục khối CT 3D. Cần làm rõ mối liên hệ giữa X-ray NVS và 3D CT reconstruction.
* **Technical Action Plan:**
  * **1-View to 3D CT Bridge:** Position 360° NVS as a regularized intermediate representation bridging 2D CXR to 3D CT reconstruction. Detail exact FDK tomographic backprojection and Beer-Lambert NeRF loss equations in `PROPOSAL.md` Section 3.
