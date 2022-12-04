# Session-based-recommenders
## Introduction
This repository was forked from the [rn5l/session-rec](https://github.com/rn5l/session-rec) repo. Our goal was to provide a script to run experiments and provide some benchmarks for different recommendation algorithms, mainly we were interested in benchmarking the Gru4Rec, GRU-PRL(from DRL3), CSRM, SR-GNN, and the NARM recommenders.We test our result on retailrocket dataset, diginetica dataset and RecSys Challange 2015 dataset.

### Contribution
All the code and experiment are implemented and conducted by chen.

### How we evaluate and compare the model performance
We evaluate the model performance by some metrics. For Gru4Rec, CSRM, SR-GNN, and the NARM we use HitRate, MRR and MAP for evaluation. For GRU-PRL we use HitRate and NDCG for evaluation.

HR@L (Hit Ratio @ L):In recommender settings, the hit ratio is simply the fraction of users for which the correct answer is included in the recommendation list of length L(Top-L).

MRR (Mean Reciprocal Rank): MRR is short for mean reciprocal rank. It is also known as average reciprocal hit ratio (ARHR).

MAP (Mean Average Precision): MAP is the mean of Average Precision. If we have the AP for each user, it is trivial just to average it over all users to calculate the MAP.

NDCG(Normalized Discounted Cumulative Gain):NDCG stands for normalized discounted cumulative gain. 

@K : top-K recommendations [Result](#Result)

The final results is shown in 


### How we set up the experiments
Check [Getting Started](#Getting-Started)
### Algorithm

<div>
<table class="table table-hover table-bordered">
    <tr>
        <th width="20%" scope="col"> Algorithm</th>
        <th width="12%" class="conf" scope="col">File</th>
        <th width="68%" class="conf" scope="col">Description</th>
    </tr>
    <tr>
        <td scope="row">Gru4Rec</td>
        <td>gru4rec.py</td>
        <td>Hidasi et al., Recurrent Neural Networks with Top-k Gains for Session-based Recommendations, CIKM 2018.<br>
        </td>
    </tr>
    <tr>
        <td scope="row">GRU-PRL</td>
        <td>GRU_PRL_RC15.py&GRU_PRL_RETAIL.py</td>
        <td>Xin et al., Rethinking Reinforcement Learning for Recommendation: A Prompt Perspective, Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval 2022.<br>
        </td>
    </tr>
    <tr>
        <td scope="row">CSRM</td>
        <td>csrm.py</td>
        <td>Wang et al., A collaborative session-based recommendation approach with parallel memory modules, SIGIR 2019.<br>
        </td>
    </tr>
    <tr>
        <td scope="row">NARM</td>
        <td>narm.py</td>
        <td>Li et al., Neural Attentive Session-based Recommendation, CIKM 2017.
        </td>
    </tr>
    <tr>
        <td scope="row">SR-GNN</td>
        <td>gnn.py</td>
        <td>Wu et al., Session-based recommendation with graph neural networks, AAAI 2019.
        </td>
    </tr>
</table>
</div>


## Result

<div>
<div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="12%" scope="col"> Recommender</th>
            <th width="16%" class="conf" scope="col">Dataset</th>
            <th width="16%" class="conf" scope="col">HitRate@20</th>
            <th width="16%" class="conf" scope="col">MRR@20</th>
            <th width="16%" class="conf" scope="col">MAP@20</th>
            <!-- <th width="16%" class="conf" scope="col">HitRate</th> -->
        </tr>
        <tr>
            <td>GRU</td>
            <td>RetailRocket</td>
            <td>0.575222
            <td>0.317998
            <td>0.305137
            </td>
        </tr>
        <tr>
            <td>GRU</td>
            <td>Diginetica</td>
            <td>0.456242
            <td>0.160353
            <td>0.145559
            </td>
        </tr>
        <tr>
            <td>SR-GNN</td>
            <td>RetailRocket</td>
            <td>0.481872
            <td>0.261930
            <td>0.250933
            </td>
        </tr>
        <tr>
            <td>SR-GNN</td>
            <td>Diginetica</td>
            <td>0.337710
            <td>0.118414
            <td>0.107449
            </td>
        </tr>
        <tr>
            <td>CSRM</td>
            <td>RetailRocket</td>
            <td>0.528208
            <td>0.280170
            <td>0.267768
            </td>
        </tr>
        <tr>
            <td>CSRM</td>
            <td>Diginetica</td>
            <td>0.392272
            <td>0.130038
            <td>0.116926
            </td>
        </tr>
        <tr>
            <td>NARM</td>
            <td>RetailRocket</td>
            <td>0.559034
            <td>0.307978
            <td>0.295425
            </td>
        </tr>
        <tr>
            <td>NARM</td>
            <td>Diginetica</td>
            <td>0.437941
            <td>0.152351
            <td>0.138072
            </td>
        </tr>
    </table>
</div>



<div>
<div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="12%" scope="col"> Recommender</th>
            <th width="16%" class="conf" scope="col">Dataset</th>
            <th width="16%" class="conf" scope="col">Purchase:HitRate@20</th>
            <th width="16%" class="conf" scope="col">Purchase:NDCG@20</th>
            <th width="16%" class="conf" scope="col">Click:HitRate@20</th>
            <th width="16%" class="conf" scope="col">Click:NDCG@20</th>
        </tr>
        <tr>
            <td>GRU_PRL</td>
            <td>RetailRocket</td>
            <td>0.642403
            <td>0.374605
            <td>0.469645
            <td>0.257799
            </td>
        </tr>
        <tr>
            <td>GRU_PRL</td>
            <td>RSC15</td>
            <td>0.555991
            <td>0.392822
            <td>0.347951
            <td>0.218077
            </td>
        </tr>
    </table>
</div>

## Getting Started

### Requirement

Linux with Python ≥ 3.6

CUDA 11.6

An NVIDIA GPU

### Start

1. Download and Install Anaconda (https://www.anaconda.com/distribution/)
2. Go to our session-based-recommenders repository
3. Run installation:

    ```
    conda env create -f env_gpu.yaml
    ```

4. Activate the conda environment: 

    ```
    conda activate srec37
    ```

5. Datasets can be downloaded from: https://drive.google.com/drive/folders/1ritDnO_Zc6DFEU6UND9C8VCisT0ETVp5?usp=sharing (retailrocket dataset, diginetica dataset) and https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z (rsc15 dataset).



Download retailrocket dataset, diginetica dataset and rsc15 dataset.

6. Unzip file and move it to data folder.

diginetica:

    ```
    mkdir -p ./data/diginetica/raw ./data/retailrocket/raw ./data/rsc15/raw
    ```


train-item-views.csv should be placed in `data/diginetica/raw`

retailrocket:

events.csv should be placed in `data/retailrocket/raw`

rsc15:

yoochoose-clicks.dat and yoochoose-buys.dat should be placed in `data/rsc15/raw`




7. Preprocess data by running a configuration with the following command:

retailrocket:

    ```
    python run_preprocessing.py conf/preprocess/session_based/window/retailrocket.yml 
    python preprocess_gru_prl_retail.py 
    python split_data_retail.py
    python trajectory_buffer_retail.py
    ```

diginetica:

    ```
    python run_preprocessing.py conf/preprocess/session_based/window/diginetica.yml
    ```

rsc15:

    ```
    python sample_data_rsc15.py 
    python merge_and_sort_rsc15.py
    python trajectory_buffer_rsc15.py
    python split_data_rsc15.py
    ```


8. Run training and evaluation

    ```
    mkdir -p ./log/GRU
    ```


retailrocket:

    ```
    THEANO_FLAGS="device=cuda0,floatX=float32" CUDA_DEVICE_ORDER=PCI_BUS_ID python run_config.py conf/exp_retailrocket_models.yml conf/out
    ```

    ```
    python GRU_PRL_RETAIL.py --data ./data/retailrocket/raw
    ```


diginetica:

    ```
    THEANO_FLAGS="device=cuda0,floatX=float32" CUDA_DEVICE_ORDER=PCI_BUS_ID python run_config.py conf/exp_diginetica_models.yml conf/out
    ```

rsc15:

    ```
    python GRU_PRL_RC15.py --data ./data/rsc15/raw
    ```


9. Check result in results directory.
<!-- Example of configuration
```
- class: emde.model.EMDE
  params: {dataset: retailrocket, alpha: 0.9, W: 0.01, bs: 256, lr: 0.004, gamma: 0.5, n_sketches: 10,
          sketch_dim: 128, hidden_size: 2986, num_epochs: 5,
          slice_absolute_codes_filenames: ['data/retailrocket/codes/slices/SessionId_iter2_dim1024',
                                          'data/retailrocket/codes/slices/SessionId_iter4_dim1024',
                                          'data/retailrocket/codes/slices/UserId_iter3_dim1024'],
          master_data_absolute_codes_filenames: ['data/retailrocket/codes/mm/property_6',
                                                'data/retailrocket/codes/mm/property_776',
                                                'data/retailrocket/codes/mm/property_839',
                                                'data/retailrocket/codes/mm/random'],
          evaluate_from_dataLoader: True
  }
  key: emde
```

`dataset` - name of dataset

`alpha` - defines time decay in history user's sketch `sketch(t2) =alpha*W^(time_diff)*sketch(t1)`

`W` - defines time decay in history user's sketch `sketch(t2) =alpha*W^(time_diff)*sketch(t1)`

`bs` - training batch size

`lr` - learning rate

`gamma` learning rate decay after each epoch

`n_sketches` sketch depth

`sketch_dim` sketch width

`hidden_size` hidden size of feed forward neural network

`num_epochs` number of epochs

`slice_absolute_codes_filenames` list of json filename with product codes, seperate filenames per slice with extension `.{slice_number}`


`master_data_absolute_codes_filenames` list of json filename with product codes, common from all slices

`evaluate_from_dataLoader` If True evalues using pytorch dataLoader else using `predict_next` method -->

