# References, Citations, and Motivations

This document compiles the scientific publications, standards, dataset repositories, and architectural motivations that influenced the design and implementation of the **HybridGAN-BWE** speech bandwidth extension and telephony restoration system.

---

## 📑 Core Academic Research Papers

### 1. HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis
* **Authors**: Jungil Kong, Jaehyeon Kim, Jaekyun Kong
* **Key Motivation**: We adopted and modified the **Multi-Period Discriminator (MPD)** and **Multi-Scale Discriminator (MSD)** groups. The design of parallel periodic 2D convolutions is crucial for capturing the multi-period pitch structures in wideband speech, while MSD helps maintain overall waveform coherence.
* **Paper Link**: [arXiv:2010.05646](https://arxiv.org/abs/2010.05646)

### 2. Parallel WaveGAN: A Fast Waveform Generation Model Based on Generative Adversarial Networks with Multi-Resolution Spectrogram Predictions
* **Authors**: Ryuichi Yamamoto, Eunwoo Song, Jae-Min Kim
* **Key Motivation**: Formed the basis for our **Multi-Resolution STFT Loss**. Combining multi-resolution spectral convergence and log magnitude losses stabilizes adversarial waveform training and prevents phase-related artifacts.
* **Paper Link**: [arXiv:1910.11480](https://arxiv.org/abs/1910.11480)

### 3. Attention Is All You Need (Cross-Attention Fusion)
* **Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
* **Key Motivation**: Guided the implementation of the **Multi-Head Cross-Attention Fusion Block** in our generator. Query-Key-Value attention is leveraged to align and fuse the temporal representations from the 1D waveform encoder with the spectral representations from the 2D spectrogram encoder.
* **Paper Link**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

### 4. NuWave & NuWave2: Neural Audio Upsampling Models
* **Authors**: Junhyeok Lee, Seungu Han
* **Key Motivation**: Provided the comparative baseline for diffusion-based bandwidth extension. We benchmarked against NuWave2's performance to highlight the real-time inference benefits (RTF) of our hybrid GAN model.
* **Paper Link**: [arXiv:2106.08558](https://arxiv.org/abs/2106.08558)

### 5. VoiceFixer: Toward General Speech Restoration with Neural Vocoder
* **Authors**: Haohe Liu, Qiuqiang Kong, Jiawen Huang, Ives Zhang, Yi Yuan, Yuxuan Wang
* **Key Motivation**: Served as a key quality benchmark. VoiceFixer showed the power of combining time-frequency networks to rebuild degraded speech, inspiring our dual-branch hybrid generator model.
* **Paper Link**: [arXiv:2109.13731](https://arxiv.org/abs/2109.13731)

### 6. AudioUNet: Speech Enhancement and Reconstruction
* **Authors**: Volkan Kılıç, A. A. Al-Naimi, et al.
* **Key Motivation**: Provided historical baseline comparisons for convolutional neural network (CNN) architectures applied to audio upsampling.
* **Paper Link**: [arXiv:1503.01800](https://arxiv.org/abs/1503.01800) (Related U-Net concepts)

---

## 🎛️ Telephony Standards & Digital Signal Processing (DSP)

### 1. ITU-T Recommendation G.711: Pulse Code Modulation (PCM) of Voice Frequencies
* **Organization**: International Telecommunication Union (ITU)
* **Key Motivation**: Standardizes logarithmic companding ($\mu$-law and A-law) to compress 14-bit or 13-bit audio samples into 8-bit symbols. We implemented differentiable/exact versions of these standards in `utils/degradation.py` to simulate digital telephony degradation.
* **Official Specification**: [ITU-T G.711 Recommendation](https://www.itu.int/rec/T-REC-G.711/en)

### 2. ETSI GSM 06.10: Full Rate Speech Transcoding
* **Organization**: European Telecommunications Standards Institute (ETSI)
* **Key Motivation**: Standardizes legacy mobile network cellular vocoding and transmission bandpass limits. We used GSM bandpass filter parameters (300 Hz – 3400 Hz) combined with 13-bit PCM quantization to simulate legacy mobile audio channels.
* **Official Specification**: [ETSI GSM 06.10 Specs](https://www.etsi.org/deliver/etsi_gts/06/0610/05_01_00_15/gts_0610v050100p.pdf)

---

## 📊 Datasets & Evaluation Tools

### 1. CSTR VCTK Corpus: English Multi-speaker Corpus for Voice Cloning Toolkit
* **Creators**: Centre for Speech Technology Research (CSTR), University of Edinburgh
* **Key Motivation**: Served as our primary wideband ground-truth dataset. Containing 110 speakers with diverse accents, it allowed training a multi-speaker BWE generator that generalizes across accents and genders.
* **Repository Link**: [Edinburgh Datashare VCTK](https://datashare.ed.ac.uk/handle/10283/2650)

### 2. NISQA: Deep Learning-Based Framework for Speech Quality Assessment
* **Creators**: Quality and Usability Lab, TU Berlin
* **Key Motivation**: We utilized the NISQA LiveTalk corpus to evaluate our model's sim-to-real generalization performance on real Skype, cellular, and VoIP network recordings.
* **Repository Link**: [GitHub - act-lab/NISQA](https://github.com/act-lab/NISQA)

---

## 🛠️ Open-Source Frameworks & Libraries
We also acknowledge the authors and maintainers of the following libraries, which form the bedrock of our software pipeline:
* **Librosa**: [librosa.org](https://librosa.org/) (Spectrogram analysis and audio file loading)
* **PyTorch**: [pytorch.org](https://pytorch.org/) (Deep learning framework, STFT computations, model optimization)
* **Gradio**: [gradio.app](https://gradio.app/) (Web interface deployment)
* **PESQ**: [github.com/ludlows/python-pesq](https://github.com/ludlows/python-pesq) (ITU-T P.862 PESQ metric utility)
* **STOI**: [github.com/mpariente/pystoi](https://github.com/mpariente/pystoi) (Short-Time Objective Intelligibility metric)
