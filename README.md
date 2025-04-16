# vulGpt
Replication package developed during the conduction of our publication on which we examined Transformer-based pre-trained models and we investigated the best practices for their utilization for the downstream task of Vulnerability Prediction in software systems.

## Overview of the overall methodology
THe high-level overview of the overall methodology that we followed in our study is illustrated in the figure belowe:

![alt text](https://github.com/iliaskaloup/vulGpt/blob/main/methodology.png?raw=true)


### Replication Package of our research work entitled "Transfer Learning for Software Vulnerability Prediction using Transformer Models"

To replicate the analysis and reproduce the results:

~~~
git clone https://github.com/iliaskaloup/vulGpt.git
~~~
and navigate to the cloned repository.

Inside the vulGpt-main folder, there are two yaml files:

• torchenv.yml file, which is the python-conda virtual environment (venv) for the experiments that utilized the PyTorch framework (all the fine-tuning scripts in directory ml/transformersFineTuning/pt)

• tfenv.yml file, which is the python-conda virtual environment (venv) for the experiments that utilized the Tensorflow framework (all the other scripts)

There are also three folders:

• bigvul for the experiments on the Big-Vul dataset 

• devign for the experiment on the Devign dataset

• reveal for the experiment on the ReVeal dataset

Each of these three folders contain four sub-folders:

• data_mining: Contains the data_mining.py file that transforms the dataset in the format that is utilized by the ML pipelines in the ml folder. The produced files are stored in the folder "data".

• data: The folder that has the raw data and the files produced by the data_mining.py.
	In the case of Big-Vul, there is no data folder initially, as the dataset is downloaded during the execution of the data_mining.py.

• ml: The folder with the ML pipelines that actually perform the experiments of the analysis. It contains:

	• The transformersFineTuning/pt/ptBert.py performs the fine-tuning of CodeBERT (choose CodeBERT variant e.g. model_variation = "microsoft/codebert-base-mlm" (e.g., for bigvul in line 90))
	
	• The transformersFineTuning/pt/ptGpt.py performs the fine-tuning of CodeGPT-2 (choose CodeGPT-2 variant e.g. model_variation = "microsoft/CodeGPT-small-py" (e.g., for bigvul in line 91))
	
		• One can also select in ptBert.py and ptGpt.py to freeze or not to freeze the pre-trained layers and therefore apply the sentence-level embeddings extraction approach or the full fine-tuning method (e.g., in case of bigvul, to extract the sentence-level embeddings just enable the commented lines (319-321 in ptBERT.py) that define the value of the "param.requires_grad")
	
	• The word_embedding/llm_embs_featExtraction_dl.py performs the embeddings extraction approach (choose embedding_algorithm e.g. embedding_algorithm = "bert" , choose model_variation e.g. model_variation = "microsoft/codebert-base-mlm" and choose userModel e.g. userModel = "bigru")
	
	• The word_embedding/gensim/train_embs.py trains on code the static word embeddings (choose w2v or ft for Word2vec or FastText method, in line 15 in case of bigvul)
	
	• The word_embedding/gensim/w2v_dl.py and word_embedding/gensim/ft_dl.py perform the word2vec and fastText embeddings approaches respectively
	
	• The bag_of_words/bow.py performs the Bag of Words experiment

• results: It is the folder where the results are stored after the successful completion of the experiments. It also contains the statistical_test.py that performs the Wilcoxon signed-rank statistical test to identify statistical significance between the results specified in the beginning of the file, e.g.

	• relative_path1 = os.path.join("codebert-base-mlm", "embeddingsExtraction", "bigru")
	
	• relative_path2 = os.path.join("codebert-base", "embeddingsExtraction", "bigru")
	
Each python file (.py) can be invoked by running the python filename.py command from the directory of each file.


### Acknowledgements

Special thanks to HuggingFace for providing the transformers libary.

Special thanks to the paper entitled "A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries" for providing the Big-Vul dataset, which was utilized as the basis for our analysis. If you use Big-Vul, please cite:

~~~
J. Fan, Y. Li, S. Wang and T. N. Nguyen, "A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries," 2020 IEEE/ACM 17th International Conference on Mining Software Repositories (MSR), Seoul, Korea, Republic of, 2020, pp. 508-512, doi: 10.1145/3379597.3387501
~~~

Special thanks to the paper entitled "Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks" for providing the Devign dataset, which was utilized during the comparison with sota graph-based vulnerability prediction models in RQ3. If you use Devign, please cite:
~~~
Zhou, Y., Liu, S., Siow, J., Du, X., & Liu, Y. (2019). Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. Advances in neural information processing systems, 32.
~~~

Special thanks to the paper entitled "Deep Learning Based Vulnerability Detection: Are We There Yet?" for providing the ReVeal dataset, which was utilized during the comparison with sota graph-based vulnerability prediction models in RQ3. If you use ReVeal, please cite:
~~~
Chakraborty, S., Krishna, R., Ding, Y., & Ray, B. (2021). Deep learning based vulnerability detection: Are we there yet?. IEEE Transactions on Software Engineering, 48(9), 3280-3296.
~~~


### Licence

[MIT License](https://github.com/iliaskaloup/vulGpt/blob/main/LICENSE)

### Citation
To cite this paper:
~~~
I. Kalouptsoglou, M. Siavvas, A. Ampatzoglou, D. Kehagias, and A. Chatzigeorgiou (2025). Transfer learning for software vulnerability prediction using Transformer models, Journal of Systems and Software (JSS), 227, 112448, doi: 10.1016/j.jss.2025.112448
~~~
