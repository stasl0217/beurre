# BEUrRE
Resources and code* for paper "Probabilistic Box Embeddings for Uncertain Knowledge Graph Reasoning"


## Install
Make sure your local environment has the following installed:

    Python3.7
    torch == 1.4.0
    wandb == 0.9.7
    
Install the dependents using:

    pip install -r requirements.txt

## Run the experiments
To run the experiments, use:

    python ./main.py --data cn15k --task mse
    
* You can switch to NL27k using `--data nl27k`
* To train and test for the ranking task, use `--task ndcg`


To test a trained model, you can use the following command:

    python ./test.py --data nl27k --task mse --model_path ./beurre-pretrained-models/nl27k-mse.pt

The pre-trained models are available here [here](https://drive.google.com/file/d/1Ai_RJEdk4H9RHYpHOzl34ZWmJ9nrzOCR/view?usp=sharing).


## Reference
Please refer to our paper. 

Xuelu Chen*, Michael Boratko*, Muhao Chen, Shib Sankar Dasgupta, Xiang Lorraine Li, Andrew McCallum. Probabilistic Box Embeddings for Uncertain Knowledge Graph Reasoning. *Proceedings of the 19th Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, 2021

\* Indicating equal contribution



    @inproceedings{chen2021boxukg,
        title={Probabilistic Box Embeddings for Uncertain Knowledge Graph Reasoning},
        author={Chen, Xuelu and Boratko, Michael and Chen, Muhao and Dasgupta, Shib Sankar and Li, Xiang Lorraine and McCallum, Andrew},
        booktitle={Proceedings of the 19th Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
        year={2021}
    }
