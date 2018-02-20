CDP4 experiment on the NRP
====================

This repository contains an experiment that connects a saliency model to a segmentation model, using the NRP as a compatibility framework, to explain how crowding and uncrowding can occur visual perception. It relies on a subset of models that were integrated to the NRP, in order to build a modular, flexible visual system. The models will be integrated in the visual system for March 2018:

![CDP4 experiment](img/experiment.png "Components of CDP4 experiment")


Installation of the experiment
-----------

**Step 1 - install the saliency model**

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


**Step 2 - install the Laminart model**

* Inside the experiment folder, simply do ``mv visual_segmentation.py $HBP/Models/brain_model``
* You can edit this file to change the input size, the number of handled orientations, and the number of segmentation layers under ``# Main parameters``


Usage
-----------

* You can edit the ``Crowding.bibi`` file to set which models are used in the experiment.
* You can edit the ``Crowding.exc`` file to set which crowding stimuli you will show to the robot.
  * Note: you have to edit this same file to use the saliency model or not (rosnode).
