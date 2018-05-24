# Ensemble Deep Learning for Skeleton-based Action Recognition using Temporal Sliding LSTM networks

This is a Tensorflow implementation of Ensemble TS-LSTM v1, v2 and v3 models from the paper [Ensemble Deep Learning for Skeleton-based Action Recognition using Temporal Sliding LSTM networks][1] and the paper 3D Human Behavior Understanding using Generalized TS-LSTM Networks. You can see the video for the paper in [Naver D2][2] or [YouTube][3] (Korean Language).

![Model architecture](ensemble_model.png)

This is also a Tensorflow implementation of Generalized Temporal Sliding LSTM (TS-LSTM) models from the paper 3D Human Behavior Understanding using Generalized TS-LSTM Networks. The generalized TS-LSTM networks consist of multiple TS-LSTM modules, and can be controlled by the hyper-parameters such as the LSTM window sizes, temporal strides and motion feature offsets of TS-LSTM modules.

![Model architecture](generalized_model.png)

## Requirements
- Python 2.7.12 (NTU)
- [Tensorflow][4] 0.11.0rc2 (NTU)
- Python 3.5.2 (UCLA & UWA)
- [Tensorflow][4] 1.4.1 (UCLA & UWA)
- Numpy

## Dataset
### [NTU RGB+D Action Recognition Dataset][5]

We found a few problems with regard to the skeleton data in NTU RGB+D Dataset.

- *Trash skeleton*
  : Sometimes the Kinect detected trash skeletons, which are misrecognized even if there is no person.
- *Skeleton index switching*
  : Also, the skeletons are swtiched when two or more skeletons are detected.
- *Primary and secondary actor*
  : In case of actions performed by two people, there is no information about a primary and a secondary actor.

To use noraml skeleton data for correct action recognition, we refined the dataset by removing trash skeletons and determining the indexes of the primary and the secondary actors.

- *Trash skeleton*
: The trash skeleton is smaller than the normal skeleton in height.
  We measured the height of skeletons and removed smaller one.
- *Skeleton index switching*
  : We calcuated the temporal distance between adjacent skeleton frames and checked whether the distance exceeded the threshold.
  When the skeleton indexes are switched, the distance suddenly increases.
- *Primary and secondary actor*
  : The primary actor is the person who is acting, and the other is the objective of the action.
  It is impossible to determine the primary actor of all action sequences, and therefore we only check the first action sequence of each setup.
  Under the assumption that the position of the actors remains unchanged in each setup, we determined the primary actor in other action sequences by using the information of the first action sequence.

We chose only normal skeleton sequences and finally provide the actor information ([Actions_01-49.txt][10], [Actions_50-60.txt][11]) with frame numbers according to above process.
The skeleton sequences in '[samples_with_missing_skeletons.txt][6]' are also removed by the process.
Also, we upload the code in Matlab, which extracts csv files from txt files provided by [ROSE Lab][7].
The codes ([make_csv_action_0149.m][8], [make_csv_action_5060.m][9]) cover two cases, Actions 1-49 and 50-60.

- Action 1-49 (One actor)
  - One skeleton (Primary actor) appears.
  - Two skeletons (Primary actor and trash skeleton) appear without skeleton index switching.
  - Two skeletons appear with skeleton index switching.
  
- Action 50-60 (Two actors)
  - Two skeletons (Primary and secondary actor) appear without skeleton index switching.
  - Two skeletons appear with skeleton index switching.
  - Three skeletons (Primary and secondary actor and trash skeleton) appear without skeleton index switching.
  - Three skeletons appear with skeleton index switching.

[1]: http://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Ensemble_Deep_Learning_ICCV_2017_paper.pdf
[2]: http://m.tv.naver.com/v/2643231
[3]: https://youtu.be/KSy7flzu4Es
[4]: https://www.tensorflow.org/install/
[5]: https://github.com/InwoongLee/NTURGB-D
[6]: https://github.com/InwoongLee/NTURGB-D/blob/master/Matlab/samples_with_missing_skeletons.txt
[7]: http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
[8]: https://github.com/InwoongLee/TS-LSTM/blob/master/make_csv_action_0149.m
[9]: https://github.com/InwoongLee/TS-LSTM/blob/master/make_csv_action_5060.m
[10]: https://github.com/InwoongLee/TS-LSTM/blob/master/Actions_01-49.txt
[11]: https://github.com/InwoongLee/TS-LSTM/blob/master/Actions_50-60.txt
