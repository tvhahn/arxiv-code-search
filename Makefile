.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
NOW_TIME := $(shell date +"%Y-%m-%d-%H%M-%S")
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = feat-store
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

environment:
ifeq (True,$(HAS_CONDA)) # assume on local
	@echo ">>> Detected conda. Assume local computer. Installing packages from yml."
	bash install_conda_local.sh
else # assume on HPC
	@echo ">>> No Conda detected. Assume on HPC."
	bash install_env_hpc.sh
	@echo ">>> venv created. Activate with source ~/arxiv/bin/activate"
endif


## Download data
download:
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/data/download_arxiv_kaggle.py
else # assume on HPC
	$(PYTHON_INTERPRETER) src/data/download_arxiv_kaggle.py
endif

## Parse entire metadata json to csv
## 
parse_json:
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/data/parse_json.py
else # assume on HPC
	sbatch src/data/parse_json_hpc.sh
endif

## Select subset of papers based on criteria
## and save to a new CSV file that will be used as a definitive index
paper_index:
ifeq (True,$(HAS_CONDA)) # assume on local
	bash src/data/make_paper_index_local.sh
else # assume on HPC
	sbatch src/data/parse_json.sh
endif


## Download papers from arxiv
download_papers:
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/data/download_papers.py --index_file_no 8
else # assume on HPC
	$(PYTHON_INTERPRETER) src/data/download_papers.py --index_file_no 8
endif


## Make Dataset
txt: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/data/make_txt.py --n_cores 6 --pdf_root_dir $(PROJECT_DIR)/data/raw/pdfs/ --index_file_no 4
else # assume on HPC
	sbatch src/data/make_txt_hpc.sh
endif


## Perform search of keywords in papers
search: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/data/search_txt.py \
		--index_file_no 4 \
		--overwrite \
		--keep_old_files \
		--max_token_len 350
else # assume on HPC
	sbatch src/data/search_txt_hpc.sh
endif


## Compile the labels from all the individual search csvs
labels: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/data/make_labels.py \
		--path_data_dir $(PROJECT_DIR)/data/ \
		--path_label_dir $(PROJECT_DIR)/data/interim/ \
		--n_cores 2 \
		--file_type ods \
		--save_name labels.csv
else # assume on HPC
	sbatch src/data/search_txt_hpc.sh
endif


## Copy labels from project_dir to scratch (only on HPC)
copy_labels: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	echo "On local compute."
else # assume on HPC
	bash src/data/copy_labels_to_scratch.sh
endif

# Download the pretrained Scibert model
# neccessary for using the BERT model on the HPC
download_pretrained_bert: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/src/models/download_pretrained_bert.py
else # assume on HPC
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/src/models/download_pretrained_bert.py
endif


## Make BERT embeddings from the label data
bert_embeddings: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/src/features/make_bert_embeddings.py \
		--proj_dir $(PROJECT_DIR) \
		--path_label_dir $(PROJECT_DIR)/data/processed/labels/labels_complete \
		--label_file_name labels_2022-08-15.csv \
		--path_emb_dir $(PROJECT_DIR)/data/processed/embeddings \
		--emb_file_name df_embeddings_2022-08-15.pkl 
else # assume on HPC
	sbatch src/models/train_model_hpc.sh $(PROJECT_DIR)
endif


## Active learning -- add predictions and probailities to label file
add_probabilities: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/src/data/add_probabilities.py \
		--proj_dir $(PROJECT_DIR) \
		--path_trained_model_dir $(PROJECT_DIR)/models/final_results_classical_2022-07-14/model_files \
		--model_name model_24667462_rf_2022-07-15-1755-09_df_embeddings_2022-07-14.pkl \
		--scaler_name scaler_24667462_rf_2022-07-15-1755-09_df_embeddings_2022-07-14.pkl \
		--path_label_dir $(PROJECT_DIR)/data/interim \
		--label_file_name labels_8.ods

else # assume on HPC
	sbatch src/models/train_model_hpc.sh $(PROJECT_DIR)
endif


## Train the deep learning model
train: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/src/models/train_model.py
else # assume on HPC
	sbatch src/models/train_model_hpc.sh $(PROJECT_DIR)
endif


# Train classical ML models through a random search
train_classical: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/models_classical/train.py \
		--save_dir_name classical_results_interim \
		--path_emb_dir $(PROJECT_DIR)/data/processed/embeddings \
		--emb_file_name df_embeddings_2022-07-11.pkl \
		--rand_search_iter 6
else # assume on HPC
	sbatch src/models_classical/train_hpc.sh $(PROJECT_DIR) $(NOW_TIME)
endif


# Compile the results of the classical models trained during the random search
compile: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/models_classical/compile.py \
	-p $(PROJECT_DIR) \
	--n_cores 6 \
	--interim_dir_name classical_results_interim \
	--final_dir_name final_results_classical
else # assume on HPC
	sbatch src/models_classical/compile_hpc.sh $(PROJECT_DIR)
endif


# Filter out the poorly performing models from the random search
filter: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/models_classical/filter.py \
		-p $(PROJECT_DIR) \
		--keep_top_n 1 \
		--save_n_figures 4 \
		--path_data_dir $(PROJECT_DIR)/data/ \
		--path_emb_dir $(PROJECT_DIR)/data/processed/embeddings \
		--emb_file_name df_embeddings_2022-07-14.pkl \
		--final_dir_name final_results_classical_2022-07-14 \
		--save_models True
else # assume on HPC
	sbatch src/models_classical/filter_hpc.sh $(PROJECT_DIR)
endif


## Make the various figures
figures: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/visualization/visualize.py
else # assume on HPC
	sbatch src/data/make_hpc_data.sh
endif


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.out" -delete


## Run unit and integration tests
test:
	$(PYTHON_INTERPRETER) -m unittest discover -s tests

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
