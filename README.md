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
Open the URL http://localhost:8501/ or any URL that this command gives you, to test the demo.


> **P/S:** When you first open the browser demo, it should take about 3-4 minutes to download all the checkpoints of the models depending on your Internet speed. Afterwards, you're free to use the demo.

We're excited to share our this demo with you! However, please keep in mind that things might not be fully polished. If you encounter any issues or have suggestions, please feel free to open an issue. Thank you!

## Run api

To run the api, please following the steps:

Install packages (can create conda environment like above to avoid potential package conflict)
```
git clone https://github.com/hoangbros03/Cinnamon2023-HaNoiTeam.git
cd Cinnamon2023-HaNoiTeam
pip install -r requirements.txt
```

Download necessary files (change python to python3 if you see the error)
```
cd api
python file_download.py
```

Run main file (it would take some time to pre-load the model)
```
python main.py
```

You can test the api by either `curl` tool or `localhost:8000/docs`

## Dockerize api (Streamlit version will be updated later)

You can also build a docker image and run the built image if interested

To build image (use sudo if your permission is denied)
```
git clone https://github.com/hoangbros03/Cinnamon2023-HaNoiTeam.git
cd Cinnamon2023-HaNoiTeam
docker build -t api-image -f ./api/Dockerfile .
```

To run image
```
docker run -p <a port outside>:8000 -it api-image
```

You can test the api by either `curl` tool or `0.0.0.0:9000/docs`

