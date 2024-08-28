# A Novel Metric for 6D Pose Estimation


## Abstract

Current state-of-the-art evaluation methods for 6D pose estimation have several significant drawbacks. Existing error metrics can produce near-zero errors for poor pose estimations and are heavily dependent on the object point cloud used, resulting in vastly different outcomes for different objects. Furthermore, false detections are not considered at all. In this paper, we conduct experiments to provide insights into how these metrics behave under isolated errors. We also introduce a novel evaluation approach with a metric independent of point clouds, making it applicable to a broader range of use cases than current metrics.

## Introduction

This repository contains the implementation of the proposed evaluation metric/score of the paper.

## Installation
Install the required dependencies listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Usage
Inside metrics.py the implementation can be found. The method named rotation_error_metric calculates the error metric for a single object. It uses the method get_rotations_for_cls to retrieves the symmetry axis for the object, the current implementation is for the objects of the YCB-Video dataset. The method dd_score calculates the score for a list of errors.
The rest is helper methods for FFB6D.
In find_symmetry_axis.py the implementation for finding symmetry axis, using PCA can be found.

## License

Specify the license under which the repository is distributed. For example:

```markdown
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

---

Feel free to modify the template to fit the specific requirements and structure of your project.