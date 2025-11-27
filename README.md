<div align="center">

<h1> DocPTBench: Benchmarking End-to-End Photographed Document Parsing and Translation </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟. </h5>

<a href="https://github.com/Topdu/DocPTBench/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/github/license/Topdu/DocPTBench"></a>
<a href='https://www.arxiv.org/abs/2511.18434'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href="https://huggingface.co/datasets/topdu/DocPTBench" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Dataset-blue"></a>
<a href="https://modelscope.cn/datasets/topdktu/DocPTBench" target="_blank"><img src="https://img.shields.io/badge/魔搭Dataset-blue"></a>

</div>


DocPTBench is **a benchmark designed specifically for real-world photographed documents**, targeting both **document parsing** and **document translation** in challenging, realistic environments.

Unlike previous benchmarks built on clean-born digital documents, DocPTBench exposes models to:

* perspective distortion
* lighting variations / shadows
* motion blur
* physical folds & wrinkles
* noise and camera artifacts

This benchmark enables rigorous evaluation of both **Document Parsing models** and **Multimodal LLMs (MLLMs)** under practical conditions.

## 📈 Highlights from the Paper

<p align="center">
  <img src="assets/overview_results_01.png" width="75%" />
  
  (a): the results of MLLMs on English (En)-started parsing (P) and translation (T) tasks; (b): the counterpart on Chinese (Zh)-started tasks; (c): the results from document parsing expert models. Ori- refers to the original digital-born document and Photographed-is its photographed version. Text- indicates that only the textual content of the document image is used as the source-language input. Alower Edit distance indicates higher parsing quality, and a higher BLEU score reflects better translation fidelity.
</p>




- 📉 **MLLMs an average parsing drops by ~18%** on photographed docs
- 📉 **Expert models drop ~25%**
- 📉 **Translation BLEU drops by ~12%**
- 🔧 **Unwarping helps**, but does not fully restore original quality
- 💡 **CoT prompting greatly reduces instruction-following failures**


---

## 🌟 Key Features

### 📷 1,381 Realistic Photographed Documents

Including both **simulated** and **real-camera** captures.

### 🌐 8 Language Pairs for Translation

**En ↔ Zh / De / Fr / Ru** and **Zh ↔ En / De / Fr / Ru**, all **human-verified**.

### 🖼 Three Document Conditions

```
Digital-Born (Original) → Photographed → Unwarping
```
<p align="center">
  <img src="assets/pipeline_01.png" width="65%" />
</p>

### 🎯 End-to-End Evaluation

Supports both:

* Parsing-only models
* Unified end-to-end MLLMs

---

## 🖼️ Example Input & Output

Refer to the appendix of the [paper](https://www.arxiv.org/abs/2511.18434).

---

## 🧪 Evaluation

### **Document Parsing**

Refer to the [parsing.md](docs/parsing.md) for evaluation details.

### **Document Translation**

Refer to the [translation.md](docs/translation.md) for evaluation details.

---

## 🧩 Supported Model Families

### 📘 Document Parsing Models

- [x] PaddleOCR-VL
- [x] MinerU2.5
- [x] dots.ocr
- [x] MonkeyOCR
- [x] DeepSeek-OCR
- [x] olmOCR and olmOCR2
- [x] Dolphin
- [x] OCRFlux
- [x] SmolDocling
- [x] Nanonets-OCR and Nanonets-OCR2
- [ ] HunyuanOCR


### 🤖 MLLMs (Closed-Source)

- [x] Gemini2.5 Pro
- [x] Qwen-VL-Max
- [x] Kimi-VL
- [x] GLM-4.5v
- [x] Doubao 1.6-v
- [ ] Gemini3 Pro


### 🔓 Open-Source Lightweight Models

- [x] Qwen3-VL-4B
- [x] Qwen2.5-VL-3B
- [x] InternVL3-2B
- [x] InternVL3.5-2B
- [ ] Qwen3-VL-235B

---

## 📚 Citation

If you use DocPTBench, please cite:

```bibtex
@misc{docptbench2025,
  title={DocPTBench: Benchmarking End-to-End Photographed Document Parsing and Translation},
  author={Yongkun Du and Pinxuan Chen and Xuye Ying and Zhineng Chen},
  year={2025},
  eprint={2511.18434},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2511.18434}
}
```

Additionally, we encourage you to cite the following papers:

```bibtex
@misc{ouyang2024omnidocbenchbenchmarkingdiversepdf,
      title={OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations}, 
      author={Linke Ouyang and Yuan Qu and Hongbin Zhou and Jiawei Zhu and Rui Zhang and Qunshu Lin and Bin Wang and Zhiyuan Zhao and Man Jiang and Xiaomeng Zhao and Jin Shi and Fan Wu and Pei Chu and Minghao Liu and Zhenxiang Li and Chao Xu and Bo Zhang and Botian Shi and Zhongying Tu and Conghui He},
      year={2024},
      eprint={2412.07626},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.07626}, 
}
```

---

## 🙏 Acknowledgments

DocPTBench is developed based on [OmniDocBench](https://github.com/opendatalab/OmniDocBench). Thanks for their awesome work!
