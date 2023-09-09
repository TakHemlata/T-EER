t-EER: Parameter-Free Tandem Evaluation Metric of Countermeasures and Biometric Comparators
===============
This repository contains our implementation of the article published in IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE-T-PAMI), "t-EER: Parameter-Free Tandem Evaluation Metric of Countermeasures and Biometric Comparators". In this work we introduce a new metric for the joint evaluation of PAD solutions operating in situ with biometric verification.

[Paper link here]()

## t-EER plot using simulated scores
![image](image_path)

## t-EER plot using real scores
![image](image_path)


##Score file preparation

Set to use either synthetic, artificial scores, or upload real scores file containining separate countermeasure (CM) and automatic speaker verification (ASV) txt score files.

1. Upload CM and ASV scores file for experiments on SASV database.

   * Prepare a score file in a plain text format
```sh
LA_0015 LA_E_1103494 bonafide target 1.0000
LA_0015 LA_E_4861467 bonafide target 1.0000
...
```

2. Upload CM and ASV scores file for experiments on ASVspoof2021 LA database.
   (Keys and metadata are available on https://www.asvspoof.org/  )

   * Prepare a score file in a plain text format
   
```sh
CM score file:
LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval -5.8546
LA_0020 LA_E_7294490 g722 loc_tx bonafide bonafide notrim eval -5.8546
...

ASV score file:
LA_0007-alaw-ita_tx  LA_E_5013670-alaw-ita_tx  alaw  ita_tx  bonafide  nontarget  notrim  eval -4.8546
LA_0008-alaw-ita_tx  LA_E_5013671-alaw-ita_tx  alaw  ita_tx  bonafide target  notrim  eval -4.8546
...
```

### To run the script:

```
python evaluate_tEER.py true false
```

## Contact
For any query regarding this repository, please contact:

- Tomi H. Kinnunen: tomi.kinnunen[at]uef[dot]fi
- Hemlata Tak: tak[at]eurecom[dot]fr

## Citation
If you use this metric in your work please use the following citations:

```bibtex

```