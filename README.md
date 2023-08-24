# Vietnamese Diacritics Restoration
## Motivation
> This is a mini project built by Hanoi team in Cinnamon Bootcamp AI 2023. We tried to research and employ Transformer and N-Gram model to solve the task add diacritical marks to a unaccented Vietnamese sentences. This presents a significant challenge when it comes to harmonizing all data into a consistent format (Vietnamese with diacritics) in order to accurately and comprehensively extract information from the data.The task of adding diacritics to Vietnamese text can be applied to many applications in LNP for Vietnamese like correcting conversation or even preprocessing step after OCR module, enabling more precise and thorough information extraction. Comprehensive information, including the complete report and presentation slides, is accessible through [this](https://drive.google.com/drive/folders/1pV0r31NDQUiklUVZ12e0_bHxqFLVLpea?usp=sharing).

## Run demo with streamlit
### Install
In the project directory, you should create conda environment and install the requirements
```bash
conda create -n demo python=3.9
```

```bash
pip install -r requirements
```

### Run demo
```bash
streamlit run app/streamlit_app.py
```
Open the URL http://localhost:8501/ to test the demo.


> **P/S:** It should take about 3-4 minutes to download all the checkpoints of the models depending on your Internet speed. Afterwards, you're free to use our demo.

We're excited to share our this demo with you! However, please keep in mind that things might not be fully polished. If you encounter any issues or have suggestions, please feel free to open an issue. Thank you!
