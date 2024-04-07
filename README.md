
## Inductive Collaborative Filtering (IDCF) open-source project by Subhadda

The codes and datasets used in the ICML'21 paper 'Towards Open-World Recommendation: An Inductive Model-Based Collaborative Filtering Approach'. There is a Chinese tutorial [Blog](https://zhuanlan.zhihu.com/p/451858601?) that introduces this work in an easy-to-follow manner.

This work proposes a new collaborative filtering approach for recsys. The new method could achieve inductive learning for new users in testing set and also help to address the cold-start problem on the user side.

![image](https://user-images.githubusercontent.com/22075007/161984853-2b697a78-d4b3-436c-8b79-0019e2bfbd59.png)

### Dependency

Python 3.8, Pytorch 1.7, Pytorch Geometric 1.6

### Download trained model and data

The trained model and preprocessed data can be downloaded by the Google Drive link shared in the repository.

You can make a directory `./data` in the root and download the data into it.

### Reproduce results

To reproduce the results in our paper (i.e., Table 2, 3, 4), you need to first download the trained model and data to corresponding folders and run the `test.py` script in each folder. Follow the steps shared in the repository.

### Run the code for training

Our model needs a two-stage training. To train the model from the beginning, you can run two scripts in order. Details are shared in the repository.

For model details, please refer to our paper. If you have any question, feel free to contact via email.

If you found the codes or datasets useful, please consider citing our paper:

```bibtex
    @inproceedings{wu2021idcf,
    title = {Towards Open-World Recommendation: An Inductive Model-Based Collaborative Filtering Approach},
    author = {QQitian Wu and Hengrui Zhang and Xiaofeng Gao and Junchi Yan and Hongyuan Zha},
    booktitle = {International Conference on Machine Learning (ICML)},
    year = {2021}
    }
```