mkdir -p resources/datasets/kidney
wget https://zenodo.org/record/5889558/files/Kidney_Chromophobe.zip?download=1 -O resources/datasets/kidney/Kidney_Chromophobe.zip
wget https://zenodo.org/record/5889558/files/Kidney_renal_clear_cell_carcinoma.zip?download=1 -O resources/datasets/kidney/Kidney_renal_clear_cell_carcinoma.zip
wget https://zenodo.org/record/5889558/files/Kidney_renal_papillary_cell_carcinoma.zip?download=1 -O resources/datasets/kidney/Kidney_renal_papillary_cell_carcinoma.zip
cd resources/datasets/kidney && unzip Kidney_Chromophobe.zip && unzip Kidney_renal_clear_cell_carcinoma.zip && unzip Kidney_renal_papillary_cell_carcinoma.zip && rm *.zip