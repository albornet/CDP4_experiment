CDP4 experiment on the NRP
====================

This repository contain the highlight CDP4 experiment on the NRP.
Here is a block diagram of the components that will be integrated in this experiment for March 2018:

![CDP4 experiment](img/experiment.png "Components of CDP4 experiment")


Installation
-----------

* Clone this folder into the ``Experiments/`` folder
* Move the brain file ``visual_segmentation.py`` in ``$HBP/Models/brain_model/``
* The following repos are needed in the `GazeboRosPackages/` folder:
  * [embodied_attention](https://github.com/HBPNeurorobotics/embodied_attention)
  * **Optinal** - [holographic](https://github.com/HBPNeurorobotics/holographic)

Don't forget to run ``catkin_make`` in your ``GazeboRosPackages/``.

Additionally, the following libraries should be installed in your platform virtual environment (``~/.opt/platform_venv``):
* keras==1.2.2
* theano==0.9.0
* scikit-image
* wget (used to download the weights/topology of the saliency network on first run)
