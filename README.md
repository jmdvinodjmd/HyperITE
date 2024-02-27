## Dynamic Inter-treatment Information Sharing for Individualized Treatment Effects Estimation
Estimation of individualized treatment effects (ITE) from observational studies is a fundamental problem in causal inference and holds significant importance across domains, including healthcare. However, limited observational datasets pose challenges in reliable ITE estimation as data have to be split among treatment groups to train an ITE learner. While information sharing among treatment groups can partially alleviate the problem, there is currently no general framework for end-to-end information sharing in ITE estimation. To tackle this problem, we propose a deep learning framework based on `\textit{soft weight sharing}' to train ITE learners, enabling \textit{dynamic end-to-end} information sharing among treatment groups. The proposed framework complements existing ITE learners, and introduces a new class of ITE learners, referred to as \textit{HyperITE}. We extend state-of-the-art ITE learners with \textit{HyperITE} versions and evaluate them on IHDP, ACIC-2016, and Twins benchmarks. Our experimental results show that the proposed framework improves ITE estimation error, with increasing effectiveness for smaller datasets.

## Usage
You can run experiments in batch using _run_script.sh_ or run an individual learner using _run_learner.py_, for example  

_python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8_

## Interested only in hypernets?
Then look for '_hn_utils.py'_ and '_hypernetworks.py'_ under moldels folder

## Paper Link
[Dynamic Inter-treatment Information Sharing for Individualized Treatment Effects Estimation](https://arxiv.org/abs/2305.15984)

## Citation
In case you find the code helpful, please consider citing our work:
```bibtex
@misc{chauhan2024dynamic,
      title={Dynamic Inter-treatment Information Sharing for Individualized Treatment Effects Estimation}, 
      author={Vinod Kumar Chauhan and Jiandong Zhou and Ghadeer Ghosheh and Soheila Molaei and David A. Clifton},
      year={2024},
      eprint={2305.15984},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
## Contact:
[Dr. Vinod Kumar Chauhan](https://sites.google.com/site/jmdvinodjmd/)
