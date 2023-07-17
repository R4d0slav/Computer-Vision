# Long-Term Tracking

Tracking objects in video sequences is a fundamental task in computer vision, with applications ranging from surveillance and autonomous navigation to augmented reality and human-computer interaction. 
While traditional trackers perform well in short-term scenarios, they often struggle to maintain accurate tracking over extended periods, particularly when faced with challenges such as occlusions, abrupt appearance changes, and target disappearance.

Long-term tracking, addresses the limitations of conventional trackers by introducing a novel approach that incorporates a re-detection phase. Specifically, SiamFC tracker is implemented and its capabilities enhanced to ensure robustness and persistence in tracking even under adverse conditions.

## SiamFC
<b>SiamFC</b>, short for Siamese Fully Convolutional Network, is a state-of-the-art tracker known for its exceptional performance in short-term tracking scenarios. 
It operates by learning a discriminative model that effectively matches the appearance of the target against a search region within subsequent frames. 
However, like many other trackers, SiamFC encounters difficulties in maintaining tracking accuracy when faced with long-term challenges, such as target disappearance or a significant drop in confidence levels.

To overcome these challenges, the functionality of the SiamFC tracker is extended by incorporating a re-detection phase. 
This phase triggers when the target disappears or when the tracker's confidence drops below a predefined threshold. 
During the re-detection phase, additional techniques and algorithms are leveraged to re-establish accurate tracking. 
By reintroducing the target in the subsequent frames, its persistence is ensured and the recovery of accurate tracking is facilitated even after prolonged occlusions or drastic appearance changes.

## How to use
- Run the tracker on a dataset:
```bash
python run_tracker.py −−dataset <path/to/dataset> −−net <path/to/network> −−results_dir <path/to/results/directory>
```
- Run evaluation:
```bash
python performance_evaluation.py −−dataset <path/to/dataset> −−results_dir <path/to/results/directory>
```
- Show tracking results:
```bash
python show_tracking.py --dataset <path/to/dataset> --results_dir <path/to/results/directory> --sequence <name/of/sequence>
```

## Example of long-term SiamFC

https://github.com/R4d0slav/Computer-Vision/assets/60989050/a61f302e-f163-470b-9d2b-81be4730638a
