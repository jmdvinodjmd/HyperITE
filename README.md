## Dynamic Inter-treatment Information Sharing for Individualized Treatment Effects Estimation
Estimation of individualized treatment effects (ITE) from observational studies is a fundamental problem in causal inference and holds significant importance across domains, including healthcare. However, limited observational datasets pose challenges in reliable ITE estimation as data have to be split among treatment groups to train an ITE learner. While information sharing among treatment groups can partially alleviate the problem, there is currently no general framework for end-to-end information sharing in ITE estimation. To tackle this problem, we propose a deep learning framework based on `\textit{soft weight sharing}' to train ITE learners, enabling \textit{dynamic end-to-end} information sharing among treatment groups. The proposed framework complements existing ITE learners, and introduces a new class of ITE learners, referred to as \textit{HyperITE}. We extend state-of-the-art ITE learners with \textit{HyperITE} versions and evaluate them on IHDP, ACIC-2016, and Twins benchmarks. Our experimental results show that the proposed framework improves ITE estimation error, with increasing effectiveness for smaller datasets.

## Usage
You can run experiments in batch using _run_script.sh_ or run an individual learner using _run_learner.py_, for example  

_python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8_

## Interested only in hypernets?
Then look for '_hn_utils.py'_ and '_hypernetworks.py'_ under moldels folder

## Paper Link
[Vinod Kumar Chauhan, Jiandong Zhou, Ghadeer Ghosheh, Soheila Molaei, David A Clifton 'Dynamic Inter-treatment Information Sharing for Individualized Treatment Effects Estimation', Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, PMLR 238:3529-3537, 2024.](https://proceedings.mlr.press/v238/kumar-chauhan24a.html)

## Citation
In case you find the code helpful, please consider citing our work:
```bibtex

@InProceedings{pmlr-v238-kumar-chauhan24a,
  title = 	 {Dynamic Inter-treatment Information Sharing for Individualized Treatment Effects Estimation },
  author =    {Chauhan, Vinod Kumar and Zhou, Jiandong and Ghosheh, Ghadeer and Molaei, Soheila and A Clifton, David},
  booktitle = {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {3529--3537},
  year = 	 {2024},
  editor = 	 {Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume = 	 {238},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--04 May},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v238/kumar-chauhan24a/kumar-chauhan24a.pdf},
  url = 	 {https://proceedings.mlr.press/v238/kumar-chauhan24a.html},
}

```
## Contact:
[Dr. Vinod Kumar Chauhan](https://sites.google.com/site/jmdvinodjmd/)
